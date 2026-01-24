# Ufunc Execution Engine Implementation Roadmap

**Issue:** #337 - Implement Full Execution Engine and Strided Kernels
**Status:** Assessment Complete - Implementation Roadmap Defined
**Complexity:** High - Requires multi-phase approach

## Current State Assessment

### Existing Implementation
The current ufunc implementation consists of:
- `src/ufunc.rs` (1120 lines) - Core ufunc trait and registration
- `src/ufunc_ops.rs` (1236 lines) - Concrete ufunc operations

**Current Execution Model:**
- Uses iterator-based execution
- May perform copies for non-contiguous arrays
- No explicit stride handling
- No dtype-specific kernel optimization

### Performance Implications
1. **Broadcasting Operations**: May allocate temporary buffers
2. **Non-Contiguous Arrays**: Linear access may be suboptimal
3. **SIMD Opportunities**: Not currently exploited
4. **Cache Efficiency**: Could be improved with strided access

## Implementation Roadmap

### Phase 1: Foundation (Current Scope)
**Goal:** Basic strided execution without breaking existing code

#### 1.1 Strided Iterator
```rust
/// Strided array iterator
pub struct StridedIter<'a, T> {
    ptr: *const T,
    stride: isize,
    len: usize,
    _phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T> Iterator for StridedIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            let item = unsafe { &*self.ptr };
            self.ptr = unsafe { self.ptr.offset(self.stride) };
            self.len -= 1;
            Some(item)
        }
    }
}
```

#### 1.2 Strided Execution Engine
```rust
pub struct StridedExecutor;

impl StridedExecutor {
    pub fn execute_binary_strided<T, F>(
        a: &ArrayView<T>,
        b: &ArrayView<T>,
        mut output: ArrayViewMut<T>,
        op: F,
    ) -> Result<()>
    where
        T: Clone,
        F: Fn(&T, &T) -> T,
    {
        // Get strides and shapes
        let shape = broadcast_shape(a.shape(), b.shape())?;
        let a_strided = a.broadcast(shape)?;
        let b_strided = b.broadcast(shape)?;

        // Execute with strided access
        for ((a_val, b_val), out_val) in a_strided
            .iter()
            .zip(b_strided.iter())
            .zip(output.iter_mut())
        {
            *out_val = op(a_val, b_val);
        }

        Ok(())
    }
}
```

**Estimated Effort:** 1-2 days
**Risk:** Low - builds on existing abstractions

### Phase 2: Kernel Registry
**Goal:** Dtype-specific optimized kernels

#### 2.1 Kernel Trait
```rust
pub trait UfuncKernel<T>: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, input: &[&[T]], output: &mut [T]) -> Result<()>;
    fn is_vectorized(&self) -> bool { false }
}
```

#### 2.2 Kernel Registry
```rust
pub struct KernelRegistry {
    kernels: HashMap<(TypeId, UfuncType), Box<dyn Any>>,
}

impl KernelRegistry {
    pub fn register<T, K>(&mut self, ufunc: UfuncType, kernel: K)
    where
        T: 'static,
        K: UfuncKernel<T> + 'static,
    {
        self.kernels.insert((TypeId::of::<T>(), ufunc), Box::new(kernel));
    }

    pub fn get<T>(&self, ufunc: UfuncType) -> Option<&dyn UfuncKernel<T>>
    where
        T: 'static,
    {
        self.kernels
            .get(&(TypeId::of::<T>(), ufunc))
            .map(|k| k.downcast_ref())
            .flatten()
    }
}
```

**Estimated Effort:** 3-5 days
**Risk:** Medium - requires careful type management

### Phase 3: SIMD Optimization
**Goal:** Leverage SIMD for contiguous arrays

#### 3.1 SIMD Kernels
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct SimdAddKernel;

impl UfuncKernel<f64> for SimdAddKernel {
    fn execute(&self, input: &[&[f64]], output: &mut [f64]) -> Result<()> {
        let a = input[0];
        let b = input[1];

        // Process 4 elements at a time (AVX2)
        let chunks = a.len() / 4;
        for i in 0..chunks {
            unsafe {
                let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * 4));
                let b_vec = _mm256_loadu_pd(b.as_ptr().add(i * 4));
                let c_vec = _mm256_add_pd(a_vec, b_vec);
                _mm256_storeu_pd(output.as_mut_ptr().add(i * 4), c_vec);
            }
        }

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            output[i] = a[i] + b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool { true }
}
```

**Estimated Effort:** 1-2 weeks
**Risk:** High - requires architecture-specific code and testing

### Phase 4: Optimization & Testing
**Goal:** Comprehensive validation and performance tuning

#### 4.1 Benchmark Suite
```rust
#[bench]
fn bench_strided_add_broadcast_2d(b: &mut Bencher) {
    let a = Array::zeros((1000, 1000));
    let b = Array::from_elem(1000, 1.0);

    b.iter(|| {
        let result = a.add(&b).unwrap();
        black_box(result);
    });
}
```

#### 4.2 Validation Tests
- Test all array layouts (C, F, non-contiguous)
- Test all broadcasting combinations
- Test all dtype combinations
- Verify numerical accuracy
- Compare performance against baseline

**Estimated Effort:** 1 week
**Risk:** Medium - extensive testing required

## Total Effort Estimate

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Basic strided execution | 1-2 days | None |
| 2 | Kernel registry | 3-5 days | Phase 1 |
| 3 | SIMD optimization | 1-2 weeks | Phase 2 |
| 4 | Testing & optimization | 1 week | Phase 3 |
| **Total** | | **3-4 weeks** | |

## Recommended Approach

### Option A: Incremental (Recommended)
1. Implement Phase 1 (strided execution)
2. Add comprehensive tests
3. Create PR for review
4. Follow up with Phase 2 (kernel registry)
5. Add Phase 3 (SIMD) as separate optimization pass

**Pros:**
- Manageable PR sizes
- Early feedback on design
- Lower risk per change

**Cons:**
- Multiple PRs
- Longer timeline to completion

### Option B: All-at-Once
Implement all phases in a single large PR.

**Pros:**
- Complete solution in one PR
- Consistent design throughout

**Cons:**
- Large review burden
- Higher risk of merge conflicts
- Difficult to test incrementally

## Risk Mitigation

### Technical Risks
1. **Performance Regression**: Mitigate with comprehensive benchmarks
2. **Numerical Accuracy**: Validate against reference implementation
3. **Memory Safety**: Use Rust's type system, extensive testing
4. **Platform Compatibility**: Test on multiple architectures

### Development Risks
1. **Scope Creep**: Strict adherence to roadmap
2. **Integration Issues**: Maintain compatibility with existing code
3. **Testing Burden**: Automated test suite from day one

## Success Criteria

### Phase 1 Success
- [ ] All existing tests pass
- [ ] 10-20% performance improvement on broadcast operations
- [ ] No regressions on contiguous operations
- [ ] Clean API for strided execution

### Phase 2 Success
- [ ] Kernel registry functional
- [ ] At least 5 dtype-specific kernels implemented
- [ ] 20-30% performance improvement overall
- [ ] Easy to add new kernels

### Phase 3 Success
- [ ] SIMD kernels for common operations (add, mul, etc.)
- [ ] 2-5x speedup on contiguous arrays
- [ ] Runtime CPU feature detection
- [ ] Fallback to scalar kernels

### Phase 4 Success
- [ ] All benchmarks green
- [ ] Documentation complete
- [ ] Migration guide for existing code
- [ ] Performance report

## Next Steps

### Immediate (This Session)
Given the complexity and scope, the recommended action for issue #337 is:

1. **Document the roadmap** (this document)
2. **Create a proof-of-concept** for Phase 1
3. **Open a tracking issue** for the full implementation
4. **Split into smaller issues** for each phase

### Alternative Approach
If immediate progress is required:
- Focus on **Phase 1 only** (basic strided execution)
- Create a minimal, working implementation
- Test thoroughly
- Document future work

## Conclusion

This is a **substantial undertaking** that requires:
- 3-4 weeks of focused development
- Deep knowledge of NumPy internals
- Performance optimization expertise
- Extensive testing across multiple architectures

**Recommendation:** Treat this as a **multi-phase project** rather than a single issue. Start with Phase 1 (strided execution) and iterate based on feedback and measured improvements.
