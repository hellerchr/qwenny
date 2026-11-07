//! Device memory for MLX's caches, which grow as their slots hold more (model.rs allocates KV in chunks on demand): what
//! a slot of `len` positions takes, against what the device has left once the weights are loaded. The server admits
//! requests and fits the context by it (Engine::cache_room, slot_bytes), as on CUDA.
use super::weights::{CONV_K, Dims, HD, LK_HEADS, LKD, LVD, NKV};

/// Device memory kept for a step's other temporaries (products, logits, compiled graphs)
pub const WORKSPACE_RESERVE: usize = 2 << 30;

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// bytes all slots together may take
    pub room: usize,
    /// a slot's DeltaNet state, whatever its length
    fixed: usize,
    /// a slot's bytes per position
    per_token: usize,
}

impl Budget {
    /// `attention`: KV caches per slot (the attention layers and the MTP's), `delta`: DeltaNet layers, `keep`: the
    /// positions a slot keeps the delta rule's inputs of (rollback)
    pub fn new(d: Dims, attention: usize, delta: usize, keep: usize, room: usize) -> Self {
        let (hv, dv, dk, lk) = (d.lv_heads as usize, LVD as usize, LKD as usize, LK_HEADS as usize);
        // f32: the end and checkpoint states, the convolution's history, the rollback inputs (kq, v, decay, beta, dot)
        let fixed = delta
            * 4
            * (2 * hv * dv * dk
                + (keep + CONV_K as usize - 1) * d.conv_width() as usize
                + keep * (2 * lk * dk + hv * dv + 2 * hv + lk));
        // int8 keys and values with an f32 scale per head
        let kv = 2 * NKV as usize * (HD as usize + 4);
        // attention over the whole prefix: the keys and values in f32, scaled, and in f16 (one layer at a time), and a
        // layer's old cache while it grows
        let temporaries = 2 * NKV as usize * HD as usize * (4 + 4 + 2) + kv;
        Budget { room, fixed, per_token: attention * kv + temporaries }
    }

    /// Device bytes a slot of `len` positions takes (0: an empty slot)
    pub fn slot_bytes(&self, len: usize) -> usize {
        if len == 0 { 0 } else { self.fixed + self.per_token * len }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_bytes_count_kv_state_and_temporaries() {
        // the 27B: 16 attention layers and the MTP's, 48 DeltaNet layers
        let d = Dims { nh: 24, lv_heads: 48, vocab: 248320 };
        let b = Budget::new(d, 17, 48, 16, 18 << 30);
        assert_eq!(b.slot_bytes(0), 0);
        assert_eq!(b.per_token, 17 * 2080 + 20480 + 2080);
        assert_eq!(b.slot_bytes(2) - b.slot_bytes(1), b.per_token);
        // a slot of 256k positions: about 15 GiB, of which the state is a fraction
        let full = b.slot_bytes(262144) as f64 / (1u64 << 30) as f64;
        assert!((14.0..16.0).contains(&full), "{full}");
        assert!(b.fixed < 1 << 30);
    }
}
