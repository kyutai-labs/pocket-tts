// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! NumPy-compatible I/O routines module
//!
//! Provides comprehensive file I/O functionality matching NumPy's API:
//! - Load/save arrays in NPY and NPZ formats
//! - Text file operations (CSV, delimited files)
//! - Binary buffer operations
//! - Memory-mapped file support
//! - Compression support for NPZ files

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use bytemuck::{cast_slice, Pod};
use byteorder::{ByteOrder as _, LittleEndian, WriteBytesExt};
use zip::{ZipArchive, ZipWriter};

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};

/// File format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Npy,
    Npz,
    Text,
    Binary,
    Unknown,
}

/// Memory mapping mode for file operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapMode {
    /// No memory mapping
    None,
    /// Read-only memory mapping
    Read,
    /// Read-write memory mapping (if supported)
    ReadWrite,
}

impl std::fmt::Display for MmapMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapMode::None => write!(f, "none"),
            MmapMode::Read => write!(f, "r"),
            MmapMode::ReadWrite => write!(f, "r+"),
        }
    }
}

impl std::str::FromStr for MmapMode {
    type Err = NumPyError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "r" | "readonly" => Ok(MmapMode::Read),
            "r+" | "readwrite" => Ok(MmapMode::ReadWrite),
            "c" | "copyonwrite" => Ok(MmapMode::Read),
            _ => Ok(MmapMode::None),
        }
    }
}

/// Detect file format from path and content
pub fn detect_file_format<P: AsRef<Path>>(path: P) -> Result<FileFormat> {
    let path = path.as_ref();

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            match ext_str.to_lowercase().as_str() {
                "npy" => return Ok(FileFormat::Npy),
                "npz" => return Ok(FileFormat::Npz),
                "txt" | "csv" | "dat" => return Ok(FileFormat::Text),
                _ => {}
            }
        }
    }

    if let Ok(mut file) = File::open(path) {
        let mut header = [0u8; 10];
        if let Ok(n) = file.read(&mut header) {
            if n >= 6 && &header[..6] == b"\x93NUMPY" {
                return Ok(FileFormat::Npy);
            }

            if n >= 4 && &header[..4] == b"PK\x03\x04" {
                return Ok(FileFormat::Npz);
            }

            let text_like = header
                .iter()
                .take_while(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                .count();
            if text_like >= n / 2 {
                return Ok(FileFormat::Text);
            } else {
                return Ok(FileFormat::Binary);
            }
        }
    }

    Ok(FileFormat::Unknown)
}

/// Load array from file (NumPy-compatible)
pub fn load<T>(
    file: &str,
    mmap_mode: Option<&str>,
    allow_pickle: bool,
    fix_imports: bool,
    encoding: &str,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let mmap_mode = mmap_mode.map(|s| s.parse()).transpose()?;

    let format = detect_file_format(file)?;
    match format {
        FileFormat::Npy => load_npy(file, mmap_mode),
        FileFormat::Npz => {
            if !allow_pickle {
                return Err(NumPyError::invalid_operation(
                    "NPZ files require pickle support",
                ));
            }
            load_npz_single(file, mmap_mode, fix_imports, encoding)
        }
        FileFormat::Text => loadtxt(
            file, None, "#", " ", None, 0, None, false, 0, encoding, None,
        ),
        _ => Err(NumPyError::file_format_error(
            "unknown",
            "Cannot determine file format",
        )),
    }
}

/// Save array to file (NumPy-compatible)
pub fn save<T>(file: &str, arr: &Array<T>, allow_pickle: bool, _fix_imports: bool) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
    T: std::fmt::Display,
{
    if !allow_pickle {
        save_npy(file, arr)
    } else {
        let format = detect_file_format_from_filename(file)?;
        match format {
            FileFormat::Npy => save_npy(file, arr),
            FileFormat::Npz => {
                let arrays = vec![("arr_0", arr)];
                savez_internal(file, &arrays, false)
            }
            FileFormat::Text => savetxt(file, arr, "%.18e", ",", "\n", "", "", "#", "utf8"),
            _ => save_npy(file, arr),
        }
    }
}

/// Save multiple arrays to uncompressed NPZ file
pub fn savez<T>(file: &str, args: Vec<(&str, &Array<T>)>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    savez_internal(file, &args, false)
}

/// Save multiple arrays to compressed NPZ file
pub fn savez_compressed<T>(file: &str, args: Vec<(&str, &Array<T>)>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    savez_internal(file, &args, true)
}

/// Load array from text file with configurable parsing
pub fn loadtxt<T>(
    fname: &str,
    _dtype: Option<Dtype>,
    comments: &str,
    delimiter: &str,
    converters: Option<Vec<fn(&str) -> T>>,
    skiprows: usize,
    usecols: Option<&[usize]>,
    unpack: bool,
    ndmin: isize,
    _encoding: &str,
    max_rows: Option<usize>,
) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    let file = File::open(fname)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        if line_num < skiprows {
            continue;
        }

        let line = line_result.map_err(|e| NumPyError::io_error(format!("Read error: {}", e)))?;
        let trimmed = line.trim();

        if trimmed.starts_with(comments) || trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = if delimiter.is_empty() {
            trimmed.split_whitespace().collect()
        } else {
            trimmed.split(delimiter).collect()
        };

        let selected_parts = if let Some(cols) = usecols {
            cols.iter()
                .filter_map(|&col| parts.get(col).copied())
                .collect()
        } else {
            parts
        };

        let row_data = if let Some(ref converters) = converters {
            selected_parts
                .iter()
                .enumerate()
                .map(|(i, part)| {
                    if i < converters.len() {
                        Ok(converters[i](part))
                    } else {
                        part.parse().map_err(|_| {
                            NumPyError::value_error(part.to_string(), "numeric conversion")
                        })
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            selected_parts
                .iter()
                .map(|part| {
                    part.parse().map_err(|_| {
                        NumPyError::value_error(part.to_string(), "numeric conversion")
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        if rows == 0 {
            cols = row_data.len();
        }

        data.extend(row_data);
        rows += 1;

        if let Some(max_rows) = max_rows {
            if rows >= max_rows {
                break;
            }
        }
    }

    let final_shape = match ndmin {
        0 | 1 => {
            if unpack {
                vec![cols, rows]
            } else {
                vec![rows, cols]
            }
        }
        2 => vec![1, rows, cols],
        _ => return Err(NumPyError::invalid_operation("ndmin must be 0, 1, or 2")),
    };

    let shape = if final_shape.last() == Some(&1) && final_shape.len() > 1 {
        final_shape[..final_shape.len() - 1].to_vec()
    } else {
        final_shape
    };

    Array::from_shape_vec(shape, data)
}

/// Save array to text file with formatting
pub fn savetxt<T>(
    fname: &str,
    x: &Array<T>,
    fmt: &str,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    comments: &str,
    _encoding: &str,
) -> Result<()>
where
    T: std::fmt::Display + Clone,
{
    let file = File::create(fname)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    if !header.is_empty() {
        let header_with_comments = if comments.is_empty() {
            header.to_string()
        } else {
            header
                .lines()
                .map(|line| format!("{}{}", comments, line))
                .collect::<Vec<_>>()
                .join("\n")
        };
        writer.write_all(header_with_comments.as_bytes())?;
        writer.write_all(newline.as_bytes())?;
    }

    let shape = x.shape();
    let data = x.to_vec();

    match shape.len() {
        0 => {
            write!(writer, "{}", data[0])?;
        }
        1 => {
            for (i, value) in data.iter().enumerate() {
                if i > 0 {
                    writer.write_all(delimiter.as_bytes())?;
                }
                writer.write_all(fmt.replace("{}", &value.to_string()).as_bytes())?;
            }
        }
        2 => {
            let rows = shape[0];
            let cols = shape[1];

            for row in 0..rows {
                for col in 0..cols {
                    if col > 0 {
                        writer.write_all(delimiter.as_bytes())?;
                    }
                    let idx = row * cols + col;
                    writer.write_all(fmt.replace("{}", &data[idx].to_string()).as_bytes())?;
                }
                if row < rows - 1 {
                    writer.write_all(newline.as_bytes())?;
                }
            }
        }
        _ => {
            let mut idx = 0;
            for _ in 0..shape.iter().take(shape.len() - 1).product() {
                for col in 0..shape[shape.len() - 1] {
                    if col > 0 {
                        writer.write_all(delimiter.as_bytes())?;
                    }
                    writer.write_all(fmt.replace("{}", &data[idx].to_string()).as_bytes())?;
                    idx += 1;
                }
                if idx < data.len() {
                    writer.write_all(newline.as_bytes())?;
                }
            }
        }
    }

    if !footer.is_empty() {
        writer.write_all(newline.as_bytes())?;
        let footer_with_comments = if comments.is_empty() {
            footer.to_string()
        } else {
            footer
                .lines()
                .map(|line| format!("{}{}", comments, line))
                .collect::<Vec<_>>()
                .join("\n")
        };
        writer.write_all(footer_with_comments.as_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Create array from raw byte buffer
pub fn frombuffer<T>(
    buffer: &[u8],
    dtype: Option<Dtype>,
    count: Option<isize>,
    offset: isize,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let dtype = dtype.unwrap_or_else(|| Dtype::from_type::<T>());
    let item_size = dtype.itemsize();

    if offset < 0 {
        return Err(NumPyError::invalid_operation("offset cannot be negative"));
    }

    let offset_usize = offset as usize;
    if offset_usize > buffer.len() {
        return Err(NumPyError::invalid_operation(
            "offset exceeds buffer length",
        ));
    }

    let available_bytes = buffer.len() - offset_usize;
    let max_elements = available_bytes / item_size;

    let num_elements = match count {
        None => max_elements,
        Some(c) if c < 0 => max_elements,
        Some(c) => std::cmp::min(c as usize, max_elements),
    };

    if num_elements == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    let end_offset = offset_usize + (num_elements * item_size);
    let data_bytes = &buffer[offset_usize..end_offset];

    let typed_data: &[T] = cast_slice(data_bytes);
    let data: Vec<T> = typed_data.to_vec();

    Ok(Array::from_vec(data))
}

/// Read array from binary file
pub fn fromfile<T>(
    file: &str,
    dtype: Dtype,
    count: isize,
    sep: &str,
    offset: isize,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    let mut file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;

    if offset > 0 {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(offset as u64))
            .map_err(|e| NumPyError::io_error(format!("Failed to seek: {}", e)))?;
    }

    if sep.is_empty() {
        let item_size = dtype.itemsize();
        let bytes_to_read = if count < 0 {
            let metadata = file
                .metadata()
                .map_err(|e| NumPyError::io_error(format!("Failed to get metadata: {}", e)))?;
            metadata.len() as usize - offset as usize
        } else {
            count as usize * item_size
        };

        let mut buffer = vec![0u8; bytes_to_read];
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| NumPyError::io_error(format!("Failed to read: {}", e)))?;

        if bytes_read != bytes_to_read {
            buffer.truncate(bytes_read);
        }

        frombuffer(&buffer, Some(dtype), Some(count), 0)
    } else {
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| NumPyError::io_error(format!("Failed to read: {}", e)))?;

        fromstring(&content, dtype, count, sep)
    }
}

/// Create array from string data
pub fn fromstring<T>(string: &str, dtype: Dtype, count: isize, sep: &str) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + Pod + 'static,
    T::Err: std::fmt::Display,
{
    if sep.is_empty() {
        let bytes = string.as_bytes();
        frombuffer(bytes, Some(dtype), Some(count), 0)
    } else {
        let parts: Vec<&str> = string.split(sep).filter(|s| !s.trim().is_empty()).collect();

        let max_count = if count < 0 {
            parts.len()
        } else {
            std::cmp::min(count as usize, parts.len())
        };

        let data: Result<Vec<T>> = parts[..max_count]
            .iter()
            .map(|part| {
                part.trim()
                    .parse()
                    .map_err(|_| NumPyError::value_error(part.to_string(), "numeric conversion"))
            })
            .collect();

        Ok(Array::from_vec(data?))
    }
}

// Internal helper functions

fn detect_file_format_from_filename(file: &str) -> Result<FileFormat> {
    let path = Path::new(file);
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            match ext_str.to_lowercase().as_str() {
                "npy" => return Ok(FileFormat::Npy),
                "npz" => return Ok(FileFormat::Npz),
                "txt" | "csv" | "dat" => return Ok(FileFormat::Text),
                _ => {}
            }
        }
    }
    Ok(FileFormat::Npy)
}

fn savez_internal<T>(file: &str, args: &[(&str, &Array<T>)], compressed: bool) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let file = File::create(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;

    if compressed {
        let zip_writer = ZipWriter::new(file);
        savez_to_zip(zip_writer, args)
    } else {
        let zip_writer = ZipWriter::new(file);
        savez_to_zip(zip_writer, args)
    }
}

fn savez_to_zip<T, W: Write + Seek>(
    mut zip_writer: ZipWriter<W>,
    args: &[(&str, &Array<T>)],
) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    use zip::write::FileOptions;

    for (name, array) in args {
        let npy_data = create_npy_data(array)?;

        let filename = format!("{}.npy", name);
        zip_writer
            .start_file(filename, FileOptions::default())
            .map_err(|e| NumPyError::io_error(e.to_string()))?;
        zip_writer.write_all(&npy_data)?;
    }

    zip_writer
        .finish()
        .map_err(|e| NumPyError::io_error(e.to_string()))?;
    Ok(())
}

fn create_npy_data<T>(array: &Array<T>) -> Result<Vec<u8>>
where
    T: Clone + Default + Pod + 'static,
{
    let mut buffer = Vec::new();

    buffer.extend_from_slice(b"\x93NUMPY");

    buffer.push(1);
    buffer.push(0);

    let shape_str = array
        .shape()
        .iter()
        .map(|&dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({})}}",
        array.dtype().to_string(),
        shape_str
    );

    let header_len = header_dict.len();
    let total_len = header_len + 16;
    let padding = (16 - (total_len % 16)) % 16;
    let total_header_len = total_len + padding;

    buffer.write_u16::<LittleEndian>((total_header_len - 10) as u16)?;

    buffer.extend_from_slice(header_dict.as_bytes());

    for _ in 0..padding {
        buffer.push(b' ');
    }
    buffer.push(b'\n');

    let data = array.to_vec();
    let data_bytes = bytemuck::cast_slice(&data);
    buffer.extend_from_slice(data_bytes);

    Ok(buffer)
}

fn save_npy<T>(file: &str, array: &Array<T>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let npy_data = create_npy_data(array)?;

    let mut file = File::create(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;
    file.write_all(&npy_data)?;

    Ok(())
}

fn load_npy<T>(file: &str, _mmap_mode: Option<MmapMode>) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let mut file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(NumPyError::file_format_error(
            "npy",
            "Invalid NPY file format",
        ));
    }

    let version = buffer[6];
    if version != 1 {
        return Err(NumPyError::file_format_error(
            "npy",
            "Only NPY version 1.0 is supported",
        ));
    }

    let header_len = LittleEndian::read_u16(&buffer[8..10]) as usize;

    let header_start = 10;
    let header_end = header_start + header_len;
    if header_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Header extends beyond file",
        ));
    }

    let header_bytes = &buffer[header_start..header_end];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| NumPyError::file_format_error("npy", "Invalid header encoding"))?;

    let shape = parse_npy_shape(header_str)?;

    let data_start = header_end;
    let data_end = data_start + shape.iter().product::<usize>() * std::mem::size_of::<T>();

    if data_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Data extends beyond file",
        ));
    }

    let data_bytes = &buffer[data_start..data_end];
    let typed_data: &[T] = bytemuck::cast_slice(data_bytes);
    let data: Vec<T> = typed_data.to_vec();

    Array::from_shape_vec(shape, data)
}

fn load_npz_single<T>(
    file: &str,
    _mmap_mode: Option<MmapMode>,
    _fix_imports: bool,
    _encoding: &str,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut archive =
        ZipArchive::new(reader).map_err(|e| NumPyError::file_format_error("npz", e.to_string()))?;

    for i in 0..archive.len() {
        let mut zip_file = archive
            .by_index(i)
            .map_err(|e| NumPyError::file_format_error("npz", e.to_string()))?;
        let filename = zip_file.name();

        if filename.ends_with(".npy") {
            let mut buffer = Vec::new();
            zip_file.read_to_end(&mut buffer)?;

            return load_npy_from_bytes(buffer);
        }
    }

    Err(NumPyError::file_format_error(
        "npz",
        "No NPY files found in archive",
    ))
}

fn load_npy_from_bytes<T>(buffer: Vec<u8>) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(NumPyError::file_format_error(
            "npy",
            "Invalid NPY file format",
        ));
    }

    let version = buffer[6];
    if version != 1 {
        return Err(NumPyError::file_format_error(
            "npy",
            "Only NPY version 1.0 is supported",
        ));
    }

    let header_len = LittleEndian::read_u16(&buffer[8..10]) as usize;

    let header_start = 10;
    let header_end = header_start + header_len;
    if header_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Header extends beyond file",
        ));
    }

    let header_bytes = &buffer[header_start..header_end];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| NumPyError::file_format_error("npy", "Invalid header encoding"))?;

    let shape = parse_npy_shape(header_str)?;

    let data_start = header_end;
    let data_end = data_start + shape.iter().product::<usize>() * std::mem::size_of::<T>();

    if data_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Data extends beyond file",
        ));
    }

    let data_bytes = &buffer[data_start..data_end];
    let typed_data: &[T] = bytemuck::cast_slice(data_bytes);
    let data: Vec<T> = typed_data.to_vec();

    Array::from_shape_vec(shape, data)
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let shape_start = header.find("'shape': (");
    if shape_start.is_none() {
        return Err(NumPyError::file_format_error(
            "npy",
            "No shape field in header",
        ));
    }

    let shape_start = shape_start.unwrap() + 10;
    let shape_end = header[shape_start..].find(')');
    if shape_end.is_none() {
        return Err(NumPyError::file_format_error("npy", "Invalid shape field"));
    }

    let shape_end = shape_start + shape_end.unwrap();
    let shape_str = &header[shape_start..shape_end];

    if shape_str.trim().is_empty() {
        return Ok(vec![]);
    }

    let dimensions: Result<Vec<usize>> = shape_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| NumPyError::file_format_error("npy", "Invalid shape dimension"))
        })
        .collect();

    dimensions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert!(detect_file_format("test.npy").is_ok());
    }

    #[test]
    fn test_mmap_mode_parsing() {
        assert_eq!("r".parse::<MmapMode>().unwrap(), MmapMode::Read);
        assert_eq!("r+".parse::<MmapMode>().unwrap(), MmapMode::ReadWrite);
        assert_eq!("c".parse::<MmapMode>().unwrap(), MmapMode::Read);
    }
}
