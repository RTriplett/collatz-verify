//! Harvest longer inverse words from the K30 positive-certificate cache and
//! compare their window credits with the existing pm_inf + mix bitvectors.
//! Usage: word_window [draws=1000000] [B=35] [mix_j=13] [cache_path] [words_output]
//! Samples are uniform window patterns, not target-weighted endpoint work.
//! Each word requires enough window odd steps to fix its ternary residue.
//! This probe changes no production verification code or GPU bitvectors.

use cv::arith::{t_iter, POW3};
use cv::lookahead::{window_dip_ext, WindowRules, START_UNKNOWN};
use cv::rules::Rules;
use cv::sample::Rng;
use rayon::prelude::*;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
#[derive(Clone, Copy, Debug)]
struct Word {
    s: u32,
    o: u32,
    b: u128,
    r: u64,
    bits: u64,
}
#[derive(Default)]
struct Counts {
    example: Option<u64>,
    improved: u64,
    matched: u64,
    reached14: u64,
    oldkeep: [u64; 16],
    newkeep: [u64; 16],
    gain: BTreeMap<i32, u64>,
    improvements_by_fw: BTreeMap<u32, u64>,
}
fn dip_pair(
    x0: u64,
    b: u32,
    wr: &WindowRules,
    dict: &[HashMap<u64, Vec<Word>>],
) -> (i32, i32, bool, u32, u32) {
    let x0 = if x0 == 0 { 1u64 << b } else { x0 };
    let mut ud = 0u64;
    let mut x = x0;
    for i in 0..b {
        if x & 1 == 0 {
            x += x / 2;
            ud |= 1 << i;
        } else {
            x = x / 2 + 1;
        }
    }
    let (mut cur, mut old, mut new, mut evens, mut fw, mut hit) =
        (0i32, 0i32, 0i32, 0u32, 0u32, false);
    let mut bestfw = 0;
    x = x0;
    for i in 0..b {
        if ud & (1 << i) != 0 {
            let a = ((ud >> i) + 1).trailing_zeros();
            if i + a + 2 < b && (ud >> (i + a)) & 3 == 0 {
                old = old.max(cur + 306);
            }
            cur -= 179;
            evens = 0;
            fw += 1;
            x += x / 2;
        } else {
            cur += 306;
            evens += 1;
            if evens % 2 == 0 && fw > 0 {
                old = old.max(cur + 179);
            }
            x = x / 2 + 1;
        }
        old = old.max(cur);
        if fw > 0 {
            let (mut xx, mut t) = (x, 0);
            while xx != 0 && xx % 3 == 0 {
                xx /= 3;
                t += 1;
            }
            old = old.max(cur + 179 * t.min(fw) as i32);
            let j = fw.min(wr.mix.as_ref().unwrap().j);
            if let Some((s, o, _)) = wr
                .mix
                .as_ref()
                .unwrap()
                .lookup(j, (x - 1) % POW3[j as usize] as u64)
            {
                old = old.max(cur + 485 * o as i32 - 306 * s as i32);
            }
        }
        new = new.max(old);
        for o in 14..=fw.min(21) {
            let r = (x - 1) % POW3[o as usize] as u64;
            if let Some(ws) = dict[o as usize].get(&r) {
                for w in ws {
                    if ((x - 1) as u128) << w.s > w.b {
                        hit = true;
                        let v = cur + 485 * w.o as i32 - 306 * w.s as i32;
                        if v > new {
                            new = v;
                            bestfw = fw;
                        }
                    }
                }
            }
        }
    }
    (old, new, hit, fw, bestfw)
}
fn audit_example(
    x0: u64,
    b: u32,
    wr: &WindowRules,
    dict: &[HashMap<u64, Vec<Word>>],
    cutoffs: &[i32; 16],
) -> serde_json::Value {
    let (old, new, _, _, _) = dip_pair(x0, b, wr, dict);
    let ci = (0..16)
        .find(|&i| old <= cutoffs[i] && new > cutoffs[i])
        .unwrap();
    let cutoff = cutoffs[ci];
    let rep = if x0 == 0 { 1u64 << b } else { x0 };
    let (mut x, mut fw, mut cur) = (rep, 0u32, 0i32);
    for j in 1..=b {
        if x & 1 == 0 {
            x += x / 2;
            fw += 1;
            cur -= 179;
        } else {
            x = x / 2 + 1;
            cur += 306;
        }
        for o in 14..=fw.min(21) {
            if let Some(ws) = dict[o as usize].get(&((x - 1) % POW3[o as usize] as u64)) {
                for w in ws {
                    let credit = cur + 485 * w.o as i32 - 306 * w.s as i32;
                    if ((x - 1) as u128) << w.s > w.b && credit > cutoff {
                        let mut checks = Vec::new();
                        for q in [0u128, 1, 2, 1000000] {
                            let start = rep as u128 - 1 + (q << b);
                            let (y, ff) = t_iter(start, j);
                            assert_eq!(ff, fw);
                            assert_eq!(y % POW3[w.o as usize], w.r as u128);
                            let p = ((y << w.s) - w.b) / POW3[w.o as usize];
                            assert!(p > 0);
                            assert_eq!(t_iter(p, w.s), (y, w.o));
                            checks.push(json!({"q":q.to_string(),"window_start":start.to_string(),"merge_y":y.to_string(),"partner_p":p.to_string()}));
                        }
                        return json!({"x0_plus_one_pattern":x0,"window_b":b,"base_dip":old,"extended_dip":new,"class_i":ci,"f_at_m62":40+ci,"cutoff":cutoff,"j":j,"fw":fw,"current_dip":cur,"word_dip":credit,"word":{"s":w.s,"o":w.o,"b":w.b.to_string(),"residue":w.r,"forward_bits":w.bits},"checks":checks});
                    }
                }
            }
        }
    }
    unreachable!()
}
fn main() {
    let t0 = Instant::now();
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    let b: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(35);
    assert!(n > 0 && (1..=39).contains(&b));
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let certpath = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("results/walkback_reuse/cache-k30/certificates-k30.bin");
    let raw = std::fs::read(certpath).unwrap();
    assert_eq!(raw.len() % 32, 0);
    let mut words = BTreeMap::new();
    let mut shapes = BTreeMap::new();
    let mut cert_by_o = BTreeMap::new();
    for rec in raw.chunks_exact(32) {
        let n0 = u64::from_le_bytes(rec[0..8].try_into().unwrap()) as u128;
        let p = u128::from_le_bytes(rec[8..24].try_into().unwrap());
        let s = u32::from_le_bytes(rec[24..28].try_into().unwrap());
        let o = u32::from_le_bytes(rec[28..32].try_into().unwrap());
        assert!((14..=21).contains(&o), "Expected the saved K30 mix13 cache");
        let (m, _) = t_iter(n0, 30);
        assert_eq!(t_iter(p, s), (m, o));
        let bb = (m << s) - POW3[o as usize] * p;
        let (mut v, mut bits, mut bc) = (p, 0u64, 0u128);
        for j in 0..s {
            if v & 1 != 0 {
                bits |= 1 << j;
                bc = 3 * bc + (1u128 << j);
                v = (3 * v + 1) / 2;
            } else {
                v /= 2;
            }
        }
        assert_eq!(bc, bb);
        assert_eq!(v, m);
        assert_eq!(bits.count_ones(), o);
        assert_eq!(bits & 1, 1);
        let r = (m % POW3[o as usize]) as u64;
        let w = Word {
            s,
            o,
            b: bb,
            r,
            bits,
        };
        words.insert((s, o, bb), w);
        *cert_by_o.entry(o).or_insert(0u64) += 1;
    }
    let mut dict: Vec<HashMap<u64, Vec<Word>>> = (0..=21).map(|_| HashMap::new()).collect();
    let mut unique_by_o = BTreeMap::new();
    let mut translated_checks = 0;
    for w in words.values() {
        *unique_by_o.entry(w.o).or_insert(0u64) += 1;
        *shapes.entry(format!("{},{}", w.s, w.o)).or_insert(0u64) += 1;
        dict[w.o as usize].entry(w.r).or_default().push(*w);
        for h in [1u128, 100, 100000] {
            let y = w.r as u128 + POW3[w.o as usize] * h;
            let numerator = y << w.s;
            if numerator > w.b {
                let p = (numerator - w.b) / POW3[w.o as usize];
                assert_eq!((numerator - w.b) % POW3[w.o as usize], 0);
                assert_eq!(t_iter(p, w.s), (y, w.o));
                translated_checks += 1;
            }
        }
    }
    let residuekeys: usize = dict.iter().map(|d| d.len()).sum();
    let mixj: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(13);
    let imp = Rules::improved(mixj);
    let wr = WindowRules {
        pm_inf: true,
        mix: imp.mix,
    };
    let seed = 0x47951ca91837u64;
    let cutoffs: [i32; 16] = std::array::from_fn(|i| 485 * (40 + i as i32) - 306 * 62);
    let chunks: Vec<Counts> = (0..n.div_ceil(4096))
        .into_par_iter()
        .map(|block| {
            let mut rng = Rng(seed ^ (block as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let mut c = Counts::default();
            for j in block * 4096..n.min((block + 1) * 4096) {
                let x = rng.next_u64() & ((1u64 << b) - 1);
                let (old, new, hit, fw, bestfw) = dip_pair(x, b, &wr, &dict);
                if j < 10000 {
                    assert_eq!(old, window_dip_ext(x, b, START_UNKNOWN, &wr));
                }
                if fw >= 14 {
                    c.reached14 += 1;
                }
                if hit {
                    c.matched += 1;
                }
                if new > old {
                    c.improved += 1;
                    *c.gain.entry(new - old).or_default() += 1;
                    *c.improvements_by_fw.entry(bestfw).or_default() += 1;
                }
                for i in 0..16 {
                    c.oldkeep[i] += (old <= cutoffs[i]) as u64;
                    c.newkeep[i] += (new <= cutoffs[i]) as u64;
                    if old <= cutoffs[i] && new > cutoffs[i] && c.example.is_none() {
                        c.example = Some(x);
                    }
                }
            }
            c
        })
        .collect();
    let mut total = Counts::default();
    for c in chunks {
        if total.example.is_none() {
            total.example = c.example;
        }
        total.improved += c.improved;
        total.matched += c.matched;
        total.reached14 += c.reached14;
        for i in 0..16 {
            total.oldkeep[i] += c.oldkeep[i];
            total.newkeep[i] += c.newkeep[i];
        }
        for (k, v) in c.gain {
            *total.gain.entry(k).or_insert(0) += v;
        }
        for (k, v) in c.improvements_by_fw {
            *total.improvements_by_fw.entry(k).or_insert(0) += v;
        }
    }
    let flips: [u64; 16] = std::array::from_fn(|i| total.oldkeep[i] - total.newkeep[i]);
    let wordlist: Vec<_> = words
        .values()
        .map(|w| json!({"s":w.s,"o":w.o,"b":w.b.to_string(),"residue":w.r,"forward_bits":w.bits}))
        .collect();
    if let Some(wordpath) = args.get(5) {
        std::fs::write(wordpath, serde_json::to_vec(&wordlist).unwrap()).unwrap();
    }
    println!(
        "{}",
        json!({"certificates":raw.len()/32,"unique_words":words.len(),"residue_keys":residuekeys,"unique_by_o":unique_by_o,"cert_by_o":cert_by_o,"shapes":shapes,"all_outside_mix13":true,"independent_translated_word_checks":translated_checks,"samples":n,"window_b":b,"base_mix_j":mixj,"seed":seed,"threads":8,"reached14":total.reached14,"matched_words":total.matched,"improved_max_dip":total.improved,"improved_fraction":total.improved as f64/n as f64,"gain_histogram":total.gain,"improvements_by_fw":total.improvements_by_fw,"m62_cutoffs":cutoffs,"flipped_fraction_of_old_kept":(0..16).map(|i|flips[i] as f64/total.oldkeep[i] as f64).collect::<Vec<_>>(),"example":total.example.map(|x|audit_example(x,b,&wr,&dict,&cutoffs)),"old_kept":total.oldkeep,"new_kept":total.newkeep,"flips":flips,"elapsed_seconds":t0.elapsed().as_secs_f64(),"caveats":["Uniform window patterns, not weighted target endpoint work or GPU timing","Extra words require o<=fw and positive partner at minimum window representative"]})
    );
}
