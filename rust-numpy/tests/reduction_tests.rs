use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_basic() {
        let arr = array![1, 2, 3, 4, 5];
        let sum = arr.sum(None, false).unwrap();
        assert_eq!(sum.size(), 1);
        assert_eq!(*sum.get(0).unwrap(), 15);
    }

    #[test]
    fn test_sum_with_axis() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];

        let sum_axis0 = arr.sum(Some(&[0]), false).unwrap();
        assert_eq!(sum_axis0.shape(), &[3]);
        assert_eq!(*sum_axis0.get(0).unwrap(), 5);
        assert_eq!(*sum_axis0.get(1).unwrap(), 7);
        assert_eq!(*sum_axis0.get(2).unwrap(), 9);

        let sum_axis1 = arr.sum(Some(&[1]), false).unwrap();
        assert_eq!(sum_axis1.shape(), &[2]);
        assert_eq!(*sum_axis1.get(0).unwrap(), 6);
        assert_eq!(*sum_axis1.get(1).unwrap(), 15);
    }

    #[test]
    fn test_sum_keepdims() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let sum_keep = arr.sum(Some(&[0]), true).unwrap();
        assert_eq!(sum_keep.shape(), &[1, 3]);
        assert_eq!(*sum_keep.get(0).unwrap(), 5);
    }

    #[test]
    fn test_product_basic() {
        let arr = array![1, 2, 3, 4];
        let prod = arr.product(None, false).unwrap();
        assert_eq!(*prod.get(0).unwrap(), 24);
    }

    #[test]
    fn test_min_basic() {
        let arr = array![5, 2, 8, 1, 3];
        let min = arr.min(None, false).unwrap();
        assert_eq!(*min.get(0).unwrap(), 1);
    }

    #[test]
    fn test_min_with_axis() {
        let arr = array2![[5, 2, 8], [1, 6, 3]];
        let min_axis0 = arr.min(Some(&[0]), false).unwrap();
        assert_eq!(min_axis0.shape(), &[3]);
        assert_eq!(*min_axis0.get(0).unwrap(), 1);
        assert_eq!(*min_axis0.get(1).unwrap(), 2);
        assert_eq!(*min_axis0.get(2).unwrap(), 3);
    }

    #[test]
    fn test_max_basic() {
        let arr = array![5, 2, 8, 1, 3];
        let max = arr.max(None, false).unwrap();
        assert_eq!(*max.get(0).unwrap(), 8);
    }

    #[test]
    fn test_max_with_axis() {
        let arr = array2![[5, 2, 8], [1, 6, 3]];
        let max_axis0 = arr.max(Some(&[0]), false).unwrap();
        assert_eq!(max_axis0.shape(), &[3]);
        assert_eq!(*max_axis0.get(0).unwrap(), 5);
        assert_eq!(*max_axis0.get(1).unwrap(), 6);
        assert_eq!(*max_axis0.get(2).unwrap(), 8);
    }

    #[test]
    fn test_mean_basic() {
        let arr = array![1, 2, 3, 4, 5];
        let mean = arr.mean(None, false).unwrap();
        assert!((*mean.get(0).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_with_axis() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let mean_axis0 = arr.mean(Some(&[0]), false).unwrap();
        assert_eq!(mean_axis0.shape(), &[3]);
        assert!((*mean_axis0.get(0).unwrap() - 2.5).abs() < 1e-10);
        assert!((*mean_axis0.get(1).unwrap() - 3.5).abs() < 1e-10);
        assert!((*mean_axis0.get(2).unwrap() - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_var_basic() {
        let arr = array![1, 2, 3, 4, 5];
        let var = arr.var(None, false, false).unwrap();
        let expected = 2.0;
        assert!((*var.get(0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_std_basic() {
        let arr = array![1, 2, 3, 4, 5];
        let std = arr.std(None, false, false).unwrap();
        let expected = 2.0_f64.sqrt();
        assert!((*std.get(0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ptp_basic() {
        let arr = array![1, 5, 2, 8, 3];
        let ptp = arr.ptp(None, false).unwrap();
        assert_eq!(*ptp.get(0).unwrap(), 7);
    }

    #[test]
    fn test_ptp_with_axis() {
        let arr = array2![[1, 5, 3], [8, 2, 6]];
        let ptp_axis0 = arr.ptp(Some(&[0]), false).unwrap();
        assert_eq!(ptp_axis0.shape(), &[3]);
        assert_eq!(*ptp_axis0.get(0).unwrap(), 7);
        assert_eq!(*ptp_axis0.get(1).unwrap(), 3);
        assert_eq!(*ptp_axis0.get(2).unwrap(), 3);
    }

    #[test]
    fn test_argmin_basic() {
        let arr = array![10, 5, 3, 8, 2];
        let argmin = arr.argmin(None).unwrap();
        assert_eq!(argmin.size(), 1);
        assert_eq!(*argmin.get(0).unwrap(), 4);
    }

    #[test]
    fn test_argmin_axis() {
        let arr = array2![[1, 5, 3], [2, 4, 6]];
        let argmin = arr.argmin(Some(1)).unwrap();
        assert_eq!(argmin.shape(), &[2]);
        assert_eq!(*argmin.get(0).unwrap(), 0);
        assert_eq!(*argmin.get(1).unwrap(), 0);
    }

    #[test]
    fn test_argmax_basic() {
        let arr = array![1, 5, 3, 8, 2];
        let argmax = arr.argmax(None).unwrap();
        assert_eq!(argmax.size(), 1);
        assert_eq!(*argmax.get(0).unwrap(), 3);
    }

    #[test]
    fn test_argmax_axis() {
        let arr = array2![[1, 5, 3], [2, 4, 6]];
        let argmax = arr.argmax(Some(1)).unwrap();
        assert_eq!(argmax.shape(), &[2]);
        assert_eq!(*argmax.get(0).unwrap(), 1);
        assert_eq!(*argmax.get(1).unwrap(), 2);
    }

    #[test]
    fn test_all_basic() {
        let arr = array![true, true, true];
        let all = arr.all(None, false).unwrap();
        assert_eq!(*all.get(0).unwrap(), true);

        let arr2 = array![true, false, true];
        let all2 = arr2.all(None, false).unwrap();
        assert_eq!(*all2.get(0).unwrap(), false);
    }

    #[test]
    fn test_all_with_axis() {
        let arr = array2![[true, true, true], [true, false, true]];
        let all_axis0 = arr.all(Some(&[0]), false).unwrap();
        assert_eq!(all_axis0.shape(), &[3]);
        assert_eq!(*all_axis0.get(0).unwrap(), true);
        assert_eq!(*all_axis0.get(1).unwrap(), false);
        assert_eq!(*all_axis0.get(2).unwrap(), true);
    }

    #[test]
    fn test_any_basic() {
        let arr = array![false, false, true];
        let any = arr.any(None, false).unwrap();
        assert_eq!(*any.get(0).unwrap(), true);

        let arr2 = array![false, false, false];
        let any2 = arr2.any(None, false).unwrap();
        assert_eq!(*any2.get(0).unwrap(), false);
    }

    #[test]
    fn test_any_with_axis() {
        let arr = array2![[false, false, false], [true, false, false]];
        let any_axis0 = arr.any(Some(&[0]), false).unwrap();
        assert_eq!(any_axis0.shape(), &[3]);
        assert_eq!(*any_axis0.get(0).unwrap(), true);
        assert_eq!(*any_axis0.get(1).unwrap(), false);
        assert_eq!(*any_axis0.get(2).unwrap(), false);
    }

    #[test]
    fn test_cumsum_basic() {
        let arr = array![1, 2, 3, 4];
        let cumsum = arr.cumsum(None).unwrap();
        assert_eq!(cumsum.shape(), &[4]);
        assert_eq!(*cumsum.get(0).unwrap(), 1);
        assert_eq!(*cumsum.get(1).unwrap(), 3);
        assert_eq!(*cumsum.get(2).unwrap(), 6);
        assert_eq!(*cumsum.get(3).unwrap(), 10);
    }

    #[test]
    fn test_cumsum_axis() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let cumsum = arr.cumsum(Some(1)).unwrap();
        assert_eq!(cumsum.shape(), &[2, 3]);
        assert_eq!(*cumsum.get(0).unwrap(), 1);
        assert_eq!(*cumsum.get(1).unwrap(), 3);
        assert_eq!(*cumsum.get(2).unwrap(), 6);
        assert_eq!(*cumsum.get(3).unwrap(), 4);
        assert_eq!(*cumsum.get(4).unwrap(), 9);
        assert_eq!(*cumsum.get(5).unwrap(), 15);
    }

    #[test]
    fn test_cumprod_basic() {
        let arr = array![1, 2, 3, 4];
        let cumprod = arr.cumprod(None).unwrap();
        assert_eq!(*cumprod.get(0).unwrap(), 1);
        assert_eq!(*cumprod.get(1).unwrap(), 2);
        assert_eq!(*cumprod.get(2).unwrap(), 6);
        assert_eq!(*cumprod.get(3).unwrap(), 24);
    }

    #[test]
    fn test_cumprod_axis() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let cumprod = arr.cumprod(Some(1)).unwrap();
        assert_eq!(cumprod.shape(), &[2, 3]);
        assert_eq!(*cumprod.get(0).unwrap(), 1);
        assert_eq!(*cumprod.get(1).unwrap(), 2);
        assert_eq!(*cumprod.get(2).unwrap(), 6);
        assert_eq!(*cumprod.get(3).unwrap(), 4);
        assert_eq!(*cumprod.get(4).unwrap(), 20);
        assert_eq!(*cumprod.get(5).unwrap(), 120);
    }

    #[test]
    fn test_sum_floating_point() {
        let arr = array![1.5, 2.5, 3.5];
        let sum = arr.sum(None, false).unwrap();
        assert!((*sum.get(0).unwrap() - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_mean_floating_point() {
        let arr = array![1.5, 2.5, 3.5];
        let mean = arr.mean(None, false).unwrap();
        assert!((*mean.get(0).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_multiple_axes() {
        let arr = array3![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];

        let sum_both_axes = arr.sum(Some(&[0, 1]), false).unwrap();
        assert_eq!(sum_both_axes.size(), 1);
        assert_eq!(*sum_both_axes.get(0).unwrap(), 36);
    }

    #[test]
    fn test_negative_axis() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let sum_neg_axis = arr.sum(Some(&[-1]), false).unwrap();
        assert_eq!(sum_neg_axis.shape(), &[2]);
        assert_eq!(*sum_neg_axis.get(0).unwrap(), 6);
        assert_eq!(*sum_neg_axis.get(1).unwrap(), 15);
    }
}
