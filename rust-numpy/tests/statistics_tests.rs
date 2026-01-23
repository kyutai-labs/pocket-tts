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
        use crate::array::Array;
        use crate::statistics::{nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar};

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
