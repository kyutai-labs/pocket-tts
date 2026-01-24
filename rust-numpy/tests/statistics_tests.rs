use numpy::statistics::{median, nanmedian, nanpercentile, nanquantile, percentile, quantile};
use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_basic() {
        let arr = array![1.0, 3.0, 2.0, 4.0];
        let median = median(&arr, None, None, false, false).unwrap();
        assert!((*median.get(0).unwrap() - 2.5_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_median_axis_keepdims() {
        let arr = array2![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let median_axis0 = median(&arr, Some(&[0]), None, false, false).unwrap();
        assert_eq!(median_axis0.shape(), &[3]);
        assert!((*median_axis0.get(0).unwrap() - 2.5_f64).abs() < 1e-10_f64);
        assert!((*median_axis0.get(1).unwrap() - 3.5_f64).abs() < 1e-10_f64);
        assert!((*median_axis0.get(2).unwrap() - 4.5_f64).abs() < 1e-10_f64);

        let median_keep = median(&arr, Some(&[1]), None, false, true).unwrap();
        assert_eq!(median_keep.shape(), &[2, 1]);
        assert!((*median_keep.get(0).unwrap() - 2.0_f64).abs() < 1e-10_f64);
        assert!((*median_keep.get(1).unwrap() - 5.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_percentile_scalar() {
        let arr = array![0.0, 10.0, 20.0, 30.0];
        let q = array![25.0];
        let result = percentile(&arr, &q, None, None, false, "linear", false, "linear").unwrap();
        assert!((*result.get(0).unwrap() - 7.5_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_percentile_multiple_q_axis() {
        let arr = array2![[1.0, 3.0], [2.0, 4.0]];
        let q = array![0.0, 50.0, 100.0];
        let result =
            percentile(&arr, &q, Some(&[0]), None, false, "linear", false, "linear").unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert!((*result.get(0).unwrap() - 1.0_f64).abs() < 1e-10_f64);
        assert!((*result.get(1).unwrap() - 3.0_f64).abs() < 1e-10_f64);
        assert!((*result.get(2).unwrap() - 1.5_f64).abs() < 1e-10_f64);
        assert!((*result.get(3).unwrap() - 3.5_f64).abs() < 1e-10_f64);
        assert!((*result.get(4).unwrap() - 2.0_f64).abs() < 1e-10_f64);
        assert!((*result.get(5).unwrap() - 4.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_quantile_basic() {
        let arr = array![0.0, 10.0, 20.0, 30.0];
        let q = array![0.25, 0.5];
        let result = quantile(&arr, &q, None, None, false, "linear", false, "linear").unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((*result.get(0).unwrap() - 7.5_f64).abs() < 1e-10_f64);
        assert!((*result.get(1).unwrap() - 15.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_nanmedian_nanpercentile() {
        let arr = array![1.0, f64::NAN, 3.0, 5.0];
        let median_val = nanmedian(&arr, None, None, false, false).unwrap();
        assert!((*median_val.get(0).unwrap() - 3.0_f64).abs() < 1e-10_f64);

        let q = array![50.0];
        let pct = nanpercentile(&arr, &q, None, None, false, "linear", false, "linear").unwrap();
        assert!((*pct.get(0).unwrap() - 3.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_nanquantile_all_nan() {
        let arr = array![f64::NAN, f64::NAN];
        let q = array![0.5];
        let result = nanquantile(&arr, &q, None, None, false, "linear", false, "linear").unwrap();
        assert!(result.get(0).unwrap().is_nan());
    }

    #[test]
    fn test_nan_aware_stats() {
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let a = Array::from_vec(data);

        // nansum: 1+2+4+5 = 12
        let sum_res = nansum(&a, None, false).unwrap();
        assert!((sum_res.get(0).unwrap() - 12.0).abs() < 1e-10);

        // nanprod: 1*2*4*5 = 40
        let prod_res = nanprod(&a, None, false).unwrap();
        assert!((prod_res.get(0).unwrap() - 40.0).abs() < 1e-10);

        // nanmean: 12 / 4 = 3
        let mean_res = nanmean(&a, None, false).unwrap();
        assert!((mean_res.get(0).unwrap() - 3.0).abs() < 1e-10);

        // nanmin: 1
        let min_res = nanmin(&a, None, false).unwrap();
        assert!((min_res.get(0).unwrap() - 1.0).abs() < 1e-10);

        // nanmax: 5
        let max_res = nanmax(&a, None, false).unwrap();
        assert!((max_res.get(0).unwrap() - 5.0).abs() < 1e-10);

        // nanvar: mean=3, ddof=0
        // ((1-3)^2 + (2-3)^2 + (4-3)^2 + (5-3)^2) / 4 = (4 + 1 + 1 + 4) / 4 = 10 / 4 = 2.5
        let var_res = nanvar(&a, None, 0, false).unwrap();
        assert!((var_res.get(0).unwrap() - 2.5).abs() < 1e-10);

        // nanstd: sqrt(2.5) ≈ 1.5811388300841898
        let std_res = nanstd(&a, None, 0, false).unwrap();
        assert!((std_res.get(0).unwrap() - 2.5f64.sqrt()).abs() < 1e-10);
    }
}

#[test]
fn test_correlate_basic() {
    use numpy::statistics::correlate;
    let a = array![1.0, 2.0, 3.0];
    let v = array![0.0, 1.0, 0.5];
    let result = correlate(&a, &v, "valid").unwrap();
    // Cross-correlation should have length len(a) + len(v) - 1 = 5
    assert_eq!(result.size(), 5);
}

#[test]
fn test_correlate_identical() {
    use numpy::statistics::correlate;
    let a = array![1.0, 2.0, 3.0];
    let v = array![1.0, 2.0, 3.0];
    let result = correlate(&a, &v, "valid").unwrap();
    // Peak correlation at the center
    let values: Vec<_> = result.iter().collect();
    let max_idx = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(max_idx, 2); // Center position
}

#[test]
fn test_nancumsum_basic() {
    use numpy::array;
    let arr = array![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
    let result = arr.nancumsum(None).unwrap();

    // NaNs treated as zero: [1, 1+0, 1+0+3, 1+0+3+0, 1+0+3+0+5] = [1, 1, 4, 4, 9]
    assert_eq!(result.get(0).unwrap(), &1.0);
    assert_eq!(result.get(1).unwrap(), &1.0);
    assert_eq!(result.get(2).unwrap(), &4.0);
    assert_eq!(result.get(3).unwrap(), &4.0);
    assert_eq!(result.get(4).unwrap(), &9.0);
}

#[test]
fn test_nancumsum_no_nan() {
    use numpy::array;
    let arr = array![1.0, 2.0, 3.0, 4.0];
    let result = arr.nancumsum(None).unwrap();

    // Same as regular cumsum: [1, 3, 6, 10]
    assert_eq!(result.get(0).unwrap(), &1.0);
    assert_eq!(result.get(1).unwrap(), &3.0);
    assert_eq!(result.get(2).unwrap(), &6.0);
    assert_eq!(result.get(3).unwrap(), &10.0);
}

#[test]
fn test_nancumsum_axis() {
    use numpy::array2;
    let arr = array2![[1.0, f64::NAN, 3.0], [f64::NAN, 5.0, 6.0]];
    let result = arr.nancumsum(Some(1)).unwrap();

    // Along axis 1, NaNs treated as zero
    // Row 0: [1, 1+0, 1+0+3] = [1, 1, 4]
    // Row 1: [0, 0+5, 0+5+6] = [0, 5, 11]
    assert_eq!(result.get(0).unwrap(), &1.0);
    assert_eq!(result.get(1).unwrap(), &1.0);
    assert_eq!(result.get(2).unwrap(), &4.0);
    assert_eq!(result.get(3).unwrap(), &0.0);
    assert_eq!(result.get(4).unwrap(), &5.0);
    assert_eq!(result.get(5).unwrap(), &11.0);
}

#[test]
fn test_nancumprod_basic() {
    use numpy::array;
    let arr = array![2.0, f64::NAN, 3.0, f64::NAN, 4.0];
    let result = arr.nancumprod(None).unwrap();

    // NaNs treated as one: [2, 2*1, 2*1*3, 2*1*3*1, 2*1*3*1*4] = [2, 2, 6, 6, 24]
    assert_eq!(result.get(0).unwrap(), &2.0);
    assert_eq!(result.get(1).unwrap(), &2.0);
    assert_eq!(result.get(2).unwrap(), &6.0);
    assert_eq!(result.get(3).unwrap(), &6.0);
    assert_eq!(result.get(4).unwrap(), &24.0);
}

#[test]
fn test_nancumprod_no_nan() {
    use numpy::array;
    let arr = array![2.0, 3.0, 4.0];
    let result = arr.nancumprod(None).unwrap();

    // Same as regular cumprod: [2, 6, 24]
    assert_eq!(result.get(0).unwrap(), &2.0);
    assert_eq!(result.get(1).unwrap(), &6.0);
    assert_eq!(result.get(2).unwrap(), &24.0);
}

#[test]
fn test_nancumprod_axis() {
    use numpy::array2;
    let arr = array2![[2.0, f64::NAN, 3.0], [f64::NAN, 5.0, 2.0]];
    let result = arr.nancumprod(Some(1)).unwrap();

    // Along axis 1, NaNs treated as one
    // Row 0: [2, 2*1, 2*1*3] = [2, 2, 6]
    // Row 1: [1, 1*5, 1*5*2] = [1, 5, 10]
    assert_eq!(result.get(0).unwrap(), &2.0);
    assert_eq!(result.get(1).unwrap(), &2.0);
    assert_eq!(result.get(2).unwrap(), &6.0);
    assert_eq!(result.get(3).unwrap(), &1.0);
    assert_eq!(result.get(4).unwrap(), &5.0);
    assert_eq!(result.get(5).unwrap(), &10.0);
}

#[test]
fn test_nancumsum_all_nan() {
    use numpy::array;
    let arr = array![f64::NAN, f64::NAN, f64::NAN];
    let result = arr.nancumsum(None).unwrap();

    // All NaNs treated as zero: [0, 0, 0]
    assert_eq!(result.get(0).unwrap(), &0.0);
    assert_eq!(result.get(1).unwrap(), &0.0);
    assert_eq!(result.get(2).unwrap(), &0.0);
}

#[test]
fn test_nancumprod_all_nan() {
    use numpy::array;
    let arr = array![f64::NAN, f64::NAN, f64::NAN];
    let result = arr.nancumprod(None).unwrap();

    // All NaNs treated as one: [1, 1, 1]
    assert_eq!(result.get(0).unwrap(), &1.0);
    assert_eq!(result.get(1).unwrap(), &1.0);
    assert_eq!(result.get(2).unwrap(), &1.0);
}
