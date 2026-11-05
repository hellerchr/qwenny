//! The MTP's draft head: the output layer for part of the vocabulary, in Marlin's layout. Its drafts come from those rows
//! only (argmax or drawn: sampling.cuh draft_sample), a fraction of the output layer's bytes per draft; the target still
//! checks every draft against the whole vocabulary, so the output is the same and only fewer drafts may be accepted.
//! Which tokens have rows, and how that changes as a server runs: draft_vocab.rs.
use crate::checkpoint::{Checkpoint, Quant};
use crate::cuda::core::{Gemm, Mat};
use crate::draft_vocab::{Adaptive, MAX_ROWS, draft_ids};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream};
use std::path::Path;
use std::sync::Arc;

/// The draft head on the GPU and its rows
pub struct DraftHead {
    pub mat: Mat,
    /// the token of each row on the device (sampled drafts map the head's columns to tokens with it)
    pub ids: CudaSlice<i32>,
    vocab: Adaptive,
    /// the output layer as stored (the rows come from it) and its zero points
    q: Quant,
    zeros: Vec<u8>,
    /// put_block's staging buffers
    stage: (CudaSlice<u32>, CudaSlice<u16>),
}

impl DraftHead {
    /// From the output layer `lm_head` (stored int4) and the draft vocabulary in `dir`, if both are there; with the
    /// rows saved for this model, if any (and room left for the vocabulary's)
    pub fn load(s: &Arc<CudaStream>, w: &Checkpoint, dir: &Path, vocab: usize) -> Result<Option<Self>> {
        if !w.is_quant("lm_head") {
            return Ok(None);
        }
        let Some(seed) = draft_ids(dir, vocab)? else { return Ok(None) };
        let q = w.quant_matrix("lm_head")?;
        // the head's tiles are Marlin int4 (learned rows are repacked in place): another width (AutoRound's 8-bit
        // lm_head) drafts over the whole output layer instead, as stored
        if q.bits != 4 {
            eprintln!("draft head: lm_head is stored in {} bits, not int4: drafts read the whole output layer", q.bits);
            return Ok(None);
        }
        let Some(zeros) = q.zero_points() else { return Ok(None) };
        let adaptive = Adaptive::load(dir, vocab, &seed)?;
        let order: Vec<usize> = adaptive.ids().iter().map(|&t| t as usize).collect();
        let mat = Mat::marlin_rows(s, &q, &zeros, &order, MAX_ROWS)?;
        let mut ids = s.alloc_zeros(MAX_ROWS)?;
        s.memcpy_htod(&adaptive.ids().iter().map(|&t| t as i32).collect::<Vec<_>>(), &mut ids.slice_mut(0..order.len()))?;
        let stage = Gemm::block_stage(s, &mat)?;
        Ok(Some(DraftHead { mat, ids, vocab: adaptive, q, zeros, stage }))
    }

    /// Take in the tokens a step commits; whether the head grew (the MTP's graphs launch it with its row count)
    pub fn learn(&mut self, s: &Arc<CudaStream>, mm: &Gemm, toks: &[u32]) -> Result<bool> {
        let valid = self.mat.valid;
        let DraftHead { mat, ids, vocab, q, zeros, stage } = self;
        vocab.learn(toks, |rows, tiles| {
            for &t in tiles {
                let order: Vec<usize> = rows[16 * t..16 * t + 16].iter().map(|&t| t as usize).collect();
                mm.put_block(s, mat, q, zeros, t, &order, stage)?;
                let tile: Vec<i32> = order.iter().map(|&t| t as i32).collect();
                s.memcpy_htod(&tile, &mut ids.slice_mut(16 * t..16 * t + 16))?;
            }
            Ok(())
        })?;
        Ok(self.mat.valid != valid)
    }
}
