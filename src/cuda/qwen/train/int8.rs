//! The int8 frozen weights of 16-bit checkpoints too large for the GPU in 16 bits (Qwen3.5-9B: 14 GB of layers in
//! bf16, 7 GB in int8).

/// 16-bit checkpoints whose layer matrices take more than this many bytes are held as int8 (symmetric, a scale per row
/// and 128 columns: the inference engine's int8)
pub const INT8_ABOVE: f64 = 12e9;
/// Columns per int8 scale
pub const GROUP: usize = 128;

/// bf16 rows [rows][cols] as int8 + 128 packed four to a u32 (little end first), a scale per row and GROUP columns
/// (max |w| / 127) and the zero point 128 per group: the trainer's `dequant` format with 8 bits
pub fn quantize(b: &[u16], cols: usize) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let rows = b.len() / cols;
    let groups = cols / GROUP;
    let mut q = vec![0u8; b.len()];
    let mut sc = vec![0f32; rows * groups];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows.max(1));
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for ((qc, sc), bc) in q.chunks_mut(per * cols).zip(sc.chunks_mut(per * groups)).zip(b.chunks(per * cols)) {
            s.spawn(move || {
                for (r, row) in bc.chunks(cols).enumerate() {
                    for g in 0..groups {
                        let vals = &row[g * GROUP..(g + 1) * GROUP];
                        let f = |x: u16| f32::from_bits((x as u32) << 16);
                        let max = vals.iter().fold(0f32, |m, &x| m.max(f(x).abs()));
                        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                        sc[r * groups + g] = scale;
                        for (c, &x) in vals.iter().enumerate() {
                            qc[r * cols + g * GROUP + c] =
                                ((f(x) / scale).round().clamp(-127.0, 127.0) as i32 + 128) as u8;
                        }
                    }
                }
            });
        }
    });
    (q, sc, vec![128u8; rows * groups])
}
