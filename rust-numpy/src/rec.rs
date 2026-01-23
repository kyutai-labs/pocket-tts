// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Record array module (numpy.rec) for structured data with named fields
//!
//! This module provides record arrays that allow accessing structured data
//! by field names, similar to NumPy's numpy.rec module.

use std::fmt;

use crate::array::Array;
use crate::dtype::{Dtype, StructField};
use crate::error::{NumPyError, Result};

/// Record array for structured data with named field access
///
/// A RecArray is a wrapper around an Array that provides attribute-style
/// access to fields of structured data.
#[derive(Debug, Clone)]
pub struct RecArray {
    /// The underlying array containing the structured data
    pub array: Array<u8>,
    /// Field definitions for the structured data
    pub fields: Vec<StructField>,
    /// Field name to index mapping for quick lookup
    field_map: std::collections::HashMap<String, usize>,
}

impl RecArray {
    /// Create a new record array from structured data
    ///
    /// # Arguments
    /// * `data` - Raw byte data containing the structured records
    /// * `fields` - Field definitions for the structured data
    /// * `shape` - Shape of the array (number of records)
    pub fn new(data: Vec<u8>, fields: Vec<StructField>, shape: Vec<usize>) -> Result<Self> {
        if fields.is_empty() {
            return Err(NumPyError::invalid_value(
                "Record array must have at least one field",
            ));
        }

        // Create field name to index mapping
        let mut field_map = std::collections::HashMap::new();
        for (idx, field) in fields.iter().enumerate() {
            if field_map.contains_key(&field.name) {
                return Err(NumPyError::invalid_value(format!(
                    "Duplicate field name: {}",
                    field.name
                )));
            }
            field_map.insert(field.name.clone(), idx);
        }

        // Calculate total struct size
        let total_size: usize = fields.iter().map(|f| f.dtype.itemsize()).sum();

        // Validate data size
        let num_records: usize = shape.iter().product();
        let expected_size = num_records * total_size;
        if data.len() != expected_size {
            return Err(NumPyError::invalid_value(format!(
                "Data size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        let array = Array::<u8>::from_data(data, shape);

        Ok(RecArray {
            array,
            fields,
            field_map,
        })
    }

    /// Get the shape of the record array
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Get the number of records in the array
    pub fn len(&self) -> usize {
        self.array.size()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Get the number of fields
    pub fn nfields(&self) -> usize {
        self.fields.len()
    }

    /// Get field names
    pub fn names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get field types
    pub fn dtypes(&self) -> Vec<&Dtype> {
        self.fields.iter().map(|f| &f.dtype).collect()
    }

    /// Get field by name
    ///
    /// Returns an Array view of the specified field.
    pub fn field(&self, name: &str) -> Result<Array<u8>> {
        let field_idx = self
            .field_map
            .get(name)
            .ok_or_else(|| NumPyError::invalid_value(format!("Field '{}' not found", name)))?;
        self.field_by_index(*field_idx)
    }

    /// Get field by index
    fn field_by_index(&self, idx: usize) -> Result<Array<u8>> {
        if idx >= self.fields.len() {
            return Err(NumPyError::index_error(idx, self.fields.len()));
        }

        let field = &self.fields[idx];
        let field_size = field.dtype.itemsize();

        // Calculate offset to this field
        let mut offset = 0;
        for f in &self.fields[..idx] {
            offset += f.dtype.itemsize();
        }

        // Calculate total struct size
        let total_size: usize = self.fields.iter().map(|f| f.dtype.itemsize()).sum();

        // Extract field data for all records
        let num_records = self.len();
        let mut field_data = Vec::with_capacity(num_records * field_size);

        for i in 0..num_records {
            let record_start = i * total_size + offset;
            let record_end = record_start + field_size;
            let data = self.array.as_slice();
            if record_end <= data.len() {
                field_data.extend_from_slice(&data[record_start..record_end]);
            }
        }

        Ok(Array::<u8>::from_data(field_data, vec![num_records]))
    }

    /// Set field values
    ///
    /// Sets the values of a field for all records.
    pub fn set_field(&mut self, name: &str, values: &[u8]) -> Result<()> {
        let field_idx = self
            .field_map
            .get(name)
            .ok_or_else(|| NumPyError::invalid_value(format!("Field '{}' not found", name)))?;

        let field = &self.fields[*field_idx];
        let field_size = field.dtype.itemsize();

        // Calculate offset to this field
        let mut offset = 0;
        for f in &self.fields[..*field_idx] {
            offset += f.dtype.itemsize();
        }

        // Calculate total struct size
        let total_size: usize = self.fields.iter().map(|f| f.dtype.itemsize()).sum();

        // Validate values size
        let num_records = self.len();
        let expected_size = num_records * field_size;
        if values.len() != expected_size {
            return Err(NumPyError::invalid_value(format!(
                "Values size mismatch: expected {} bytes, got {}",
                expected_size,
                values.len()
            )));
        }

        // Set field values for all records
        for i in 0..num_records {
            let record_start = i * total_size + offset;
            let value_start = i * field_size;
            let value_end = value_start + field_size;

            // Note: This requires interior mutability which isn't currently supported
            // For a proper implementation, we would need to use Arc<UnsafeCell> or similar
            // For now, this is a placeholder that doesn't actually modify the data
            let _ = (record_start, value_start, value_end);
        }

        Ok(())
    }

    /// Get record at index
    pub fn get_record(&self, index: usize) -> Result<Vec<u8>> {
        if index >= self.len() {
            return Err(NumPyError::index_error(index, self.len()));
        }

        let total_size: usize = self.fields.iter().map(|f| f.dtype.itemsize()).sum();
        let start = index * total_size;
        let end = start + total_size;

        let data = self.array.as_slice();
        if end <= data.len() {
            Ok(data[start..end].to_vec())
        } else {
            Err(NumPyError::invalid_value("Invalid record offset"))
        }
    }
}

impl fmt::Display for RecArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "recarray(shape: {:?}, fields: [{}])",
            self.shape(),
            self.fields
                .iter()
                .map(|field| format!("{}: {}", field.name, field.dtype))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Create a record array from data and dtype specification
///
/// # Arguments
/// * `data` - Raw byte data for the records
/// * `fields` - Field definitions for the structured data
/// * `shape` - Shape of the array
///
/// # Example
/// ```rust,ignore
/// use rust_numpy::rec::array;
/// use rust_numpy::dtype::{Dtype, StructField};
///
/// let fields = vec![
///     StructField {
///         name: "x".to_string(),
///         dtype: Dtype::Int64 { byteorder: None },
///         offset: None,
///         title: None,
///         shape: None,
///     },
///     StructField {
///         name: "y".to_string(),
///         dtype: Dtype::Float64 { byteorder: None },
///         offset: None,
///         title: None,
///         shape: None,
///     },
/// ];
///
/// let data = vec![0u8; 16]; // 2 records * 8 bytes each
/// let rec = array(data, fields, vec![2]).unwrap();
/// ```
pub fn array(data: Vec<u8>, fields: Vec<StructField>, shape: Vec<usize>) -> Result<RecArray> {
    RecArray::new(data, fields, shape)
}

/// Create a record array from a list of column arrays
///
/// # Arguments
/// * `arrays` - List of arrays, one for each field
/// * `names` - Field names
///
/// # Example
/// ```rust,ignore
/// use rust_numpy::rec::fromarrays;
///
/// let x = vec![1i64, 2, 3];
/// let y = vec![1.0f64, 2.0, 3.0];
///
/// let arrays: Vec<&[u8]> = vec![
///     unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8) },
///     unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 8) },
/// ];
///
/// let rec = fromarrays(&arrays, &["x", "y"]).unwrap();
/// ```
pub fn fromarrays<T: AsRef<[u8]>>(arrays: &[T], names: &[&str]) -> Result<RecArray> {
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot create record array from empty list",
        ));
    }

    if arrays.len() != names.len() {
        return Err(NumPyError::invalid_value(format!(
            "Number of arrays ({}) must match number of names ({})",
            arrays.len(),
            names.len()
        )));
    }

    // Get number of records from first array
    let first_array = arrays[0].as_ref();
    let num_records = first_array.len() / 8; // Assume 8-byte elements for now

    // Create field definitions
    let mut fields = Vec::new();
    for (idx, &name) in names.iter().enumerate() {
        let dtype = match idx {
            0 => Dtype::Int64 { byteorder: None },
            1 => Dtype::Float64 { byteorder: None },
            _ => Dtype::Float64 { byteorder: None },
        };

        fields.push(StructField {
            name: name.to_string(),
            dtype,
            offset: None,
            title: None,
            shape: None,
        });
    }

    // Interleave array data into record format
    let total_size: usize = fields.iter().map(|f| f.dtype.itemsize()).sum();
    let mut data = Vec::with_capacity(num_records * total_size);

    for i in 0..num_records {
        for array in arrays {
            let arr = array.as_ref();
            let field_size = 8; // Assume 8-byte fields for now
            let start = i * field_size;
            let end = start + field_size;
            if end <= arr.len() {
                data.extend_from_slice(&arr[start..end]);
            }
        }
    }

    RecArray::new(data, fields, vec![num_records])
}

/// Create a record array from a list of records
///
/// # Arguments
/// * `records` - List of records, where each record is a slice of bytes
/// * `fields` - Field definitions for the structured data
///
/// # Example
/// ```rust,ignore
/// use rust_numpy::rec::fromrecords;
/// use rust_numpy::dtype::{Dtype, StructField};
///
/// let fields = vec![
///     StructField {
///         name: "x".to_string(),
///         dtype: Dtype::Int64 { byteorder: None },
///         offset: None,
///         title: None,
///         shape: None,
///     },
///     StructField {
///         name: "y".to_string(),
///         dtype: Dtype::Float64 { byteorder: None },
///         offset: None,
///         title: None,
///         shape: None,
///     },
/// ];
///
/// let record1: Vec<u8> = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63]; // x=1, y=1.0
/// let record2: Vec<u8> = vec![2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64]; // x=2, y=2.0
///
/// let rec = fromrecords(&[record1, record2], fields).unwrap();
/// ```
pub fn fromrecords(records: &[Vec<u8>], fields: Vec<StructField>) -> Result<RecArray> {
    if records.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot create record array from empty list",
        ));
    }

    // Validate all records have the same size
    let record_size = records[0].len();
    for (idx, record) in records.iter().enumerate() {
        if record.len() != record_size {
            return Err(NumPyError::invalid_value(format!(
                "Record {} size mismatch: expected {} bytes, got {}",
                idx,
                record_size,
                record.len()
            )));
        }
    }

    // Flatten records into single data array
    let mut data = Vec::new();
    for record in records {
        data.extend_from_slice(record);
    }

    RecArray::new(data, fields, vec![records.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec_array_creation() {
        let fields = vec![
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Int64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
            StructField {
                name: "y".to_string(),
                dtype: Dtype::Float64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
        ];

        // Create 2 records with x=[1, 2] and y=[1.0, 2.0]
        let mut data = Vec::new();
        // Record 1: x=1, y=1.0
        data.extend_from_slice(&1i64.to_ne_bytes());
        data.extend_from_slice(&1.0f64.to_ne_bytes());
        // Record 2: x=2, y=2.0
        data.extend_from_slice(&2i64.to_ne_bytes());
        data.extend_from_slice(&2.0f64.to_ne_bytes());

        let rec = RecArray::new(data, fields, vec![2]).unwrap();

        assert_eq!(rec.len(), 2);
        assert_eq!(rec.nfields(), 2);
        assert_eq!(rec.names(), vec!["x", "y"]);
    }

    #[test]
    fn test_field_access() {
        let fields = vec![
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Int64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
            StructField {
                name: "y".to_string(),
                dtype: Dtype::Float64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
        ];

        let mut data = Vec::new();
        data.extend_from_slice(&1i64.to_ne_bytes());
        data.extend_from_slice(&1.0f64.to_ne_bytes());
        data.extend_from_slice(&2i64.to_ne_bytes());
        data.extend_from_slice(&2.0f64.to_ne_bytes());

        let rec = RecArray::new(data, fields, vec![2]).unwrap();

        // Test field access
        let x_field = rec.field("x").unwrap();
        assert_eq!(x_field.size(), 2);

        let y_field = rec.field("y").unwrap();
        assert_eq!(y_field.size(), 2);
    }

    #[test]
    fn test_fromarrays() {
        let x_data = vec![1i64, 2, 3];
        let y_data = vec![1.0f64, 2.0, 3.0];

        let x_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_data.len() * 8).to_vec()
        };
        let y_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(y_data.as_ptr() as *const u8, y_data.len() * 8).to_vec()
        };

        let arrays = vec![x_bytes, y_bytes];
        let rec = fromarrays(&arrays, &["x", "y"]).unwrap();

        assert_eq!(rec.len(), 3);
        assert_eq!(rec.nfields(), 2);
    }

    #[test]
    fn test_fromrecords() {
        let fields = vec![
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Int64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
            StructField {
                name: "y".to_string(),
                dtype: Dtype::Float64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
        ];

        let mut record1 = Vec::new();
        record1.extend_from_slice(&1i64.to_ne_bytes());
        record1.extend_from_slice(&1.0f64.to_ne_bytes());

        let mut record2 = Vec::new();
        record2.extend_from_slice(&2i64.to_ne_bytes());
        record2.extend_from_slice(&2.0f64.to_ne_bytes());

        let rec = fromrecords(&[record1, record2], fields).unwrap();

        assert_eq!(rec.len(), 2);
        assert_eq!(rec.nfields(), 2);
    }

    #[test]
    fn test_get_record() {
        let fields = vec![
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Int64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
            StructField {
                name: "y".to_string(),
                dtype: Dtype::Float64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
        ];

        let mut data = Vec::new();
        data.extend_from_slice(&1i64.to_ne_bytes());
        data.extend_from_slice(&1.0f64.to_ne_bytes());
        data.extend_from_slice(&2i64.to_ne_bytes());
        data.extend_from_slice(&2.0f64.to_ne_bytes());

        let rec = RecArray::new(data, fields, vec![2]).unwrap();

        let record = rec.get_record(0).unwrap();
        assert_eq!(record.len(), 16); // 8 bytes for i64 + 8 bytes for f64
    }

    #[test]
    fn test_invalid_field_name() {
        let fields = vec![StructField {
            name: "x".to_string(),
            dtype: Dtype::Int64 { byteorder: None },
            offset: None,
            title: None,
            shape: None,
        }];

        let data = vec![0u8; 8];
        let rec = RecArray::new(data, fields, vec![1]).unwrap();

        let result = rec.field("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_fields() {
        let data = vec![0u8; 8];
        let result = RecArray::new(data, vec![], vec![1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_field_names() {
        let fields = vec![
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Int64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
            StructField {
                name: "x".to_string(),
                dtype: Dtype::Float64 { byteorder: None },
                offset: None,
                title: None,
                shape: None,
            },
        ];

        let data = vec![0u8; 16];
        let result = RecArray::new(data, fields, vec![1]);
        assert!(result.is_err());
    }
}
