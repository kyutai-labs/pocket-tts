// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Datetime support functions

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use chrono::DateTime;
use ndarray::Array1;
use num_traits::{FromPrimitive, ToPrimitive, Zero};

#[derive(Debug, Clone)]
pub enum DateTimeUnit {
    Years,
    Months,
    Weeks,
    Days,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

#[derive(Debug, Clone)]
pub struct BusDayCalendar {
    pub weekmask: [bool; 7],
    pub holidays: Array1<i64>,
}

impl BusDayCalendar {
    pub fn new(weekmask: [bool; 7], holidays: Array1<i64>) -> Self {
        Self { weekmask, holidays }
    }

    pub fn default() -> Self {
        Self {
            weekmask: [false, true, true, true, true, true, false],
            holidays: Array1::from_vec(vec![]),
        }
    }
}

pub fn datetime_data<T>(dtype: Dtype) -> Result<(String, String, i32), NumPyError>
where
    T: FromPrimitive + Zero,
{
    let (unit_name, unit_code, _step) = match dtype {
        Dtype::Datetime64(ref unit) => {
            let unit_str = unit.as_str().to_string();
            match unit_str.as_str() {
                "Y" => ("Y", "years", 1),
                "M" => ("M", "months", 1),
                "W" => ("W", "weeks", 1),
                "D" => ("D", "days", 1),
                "h" => ("h", "hours", 3600),
                "m" => ("m", "minutes", 60),
                "s" => ("s", "seconds", 1),
                "ms" => ("ms", "milliseconds", 1),
                "us" => ("us", "microseconds", 1),
                "ns" => ("ns", "nanoseconds", 1),
                _ => return Err(NumPyError::invalid_dtype(format!("{:?}", dtype))),
            }
        },
        _ => return Err(NumPyError::invalid_dtype(format!("{:?}", dtype))),
    };

    let step = match unit_code {
        "years" => 31536000,
        "months" => 2592000,
        "weeks" => 604800,
        "days" => 86400,
        "hours" => 3600,
        "minutes" => 60,
        "seconds" => 1,
        "milliseconds" => 1,
        "microseconds" => 1,
        "nanoseconds" => 1,
        _ => 1,
    };

    Ok((unit_name.to_string(), unit_code.to_string(), step))
}

pub fn busday_count<T>(
    begindates: &Array<T>,
    enddates: &Array<T>,
    weekmask: &str,
    holidays: Option<&Array<T>>,
    busdaycal: Option<BusDayCalendar>,
    _out: Option<&mut Array<T>>,
) -> Result<Array<T>, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Zero + Copy + std::fmt::Debug + Default + 'static,
{
    if begindates.shape() != enddates.shape() {
        return Err(NumPyError::shape_mismatch(
            begindates.shape().to_vec(),
            enddates.shape().to_vec(),
        ));
    }

    let weekmask_array = parse_weekmask(weekmask)?;
    let calendar = busdaycal.unwrap_or_else(|| {
        BusDayCalendar::new(
            weekmask_array,
            holidays.map_or(Array1::from_vec(vec![]), |h| {
                Array1::from_iter(h.data().iter().map(|x| x.to_i64().unwrap_or(0)))
            }),
        )
    });

    let mut result_data = Vec::with_capacity(begindates.size());

    for i in 0..begindates.size() {
        let begin_date = begindates.get_linear(i).copied().unwrap_or(T::zero());
        let end_date = enddates.get_linear(i).copied().unwrap_or(T::zero());

        let count = count_business_days(begin_date, end_date, &calendar)?;
        result_data.push(T::from_isize(count as isize).unwrap_or(T::zero()));
    }

    Ok(Array::from_data(result_data, begindates.shape().to_vec()))
}

pub fn busday_offset<T>(
    dates: &Array<T>,
    offsets: &Array<T>,
    roll: &str,
    weekmask: &str,
    holidays: Option<&Array<T>>,
    busdaycal: Option<BusDayCalendar>,
    out: Option<&mut Array<T>>,
) -> Result<Array<T>, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Zero + Copy + std::fmt::Debug + Default + 'static,
{
    if dates.shape() != offsets.shape() {
        return Err(NumPyError::shape_mismatch(
            dates.shape().to_vec(),
            offsets.shape().to_vec(),
        ));
    }

    let weekmask_array = parse_weekmask(weekmask)?;
    let calendar = busdaycal.unwrap_or_else(|| {
        BusDayCalendar::new(
            weekmask_array,
            holidays.map_or(Array1::from_vec(vec![]), |h| {
                Array1::from_iter(h.data().iter().map(|x| x.to_i64().unwrap_or(0)))
            }),
        )
    });

    let mut result = match out {
        Some(out_arr) => out_arr.to_owned(),
        None => Array::from_shape_vec(dates.shape().to_vec(), vec![T::zero(); dates.size()])?,
    };

    for i in 0..dates.size() {
        let date = dates.get_linear(i).copied().unwrap_or(T::zero());
        let offset = offsets.get_linear(i).copied().unwrap_or(T::zero());

        let new_date = apply_business_day_offset(date, offset, roll, &calendar)?;
        result.set_linear(i, new_date);
    }

    Ok(result)
}

pub fn is_busday<T>(
    dates: &Array<T>,
    weekmask: &str,
    holidays: Option<&Array<T>>,
    busdaycal: Option<BusDayCalendar>,
) -> Result<Array<bool>, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Zero + Copy + std::fmt::Debug + 'static,
{
    let weekmask_array = parse_weekmask(weekmask)?;
    let calendar = busdaycal.unwrap_or_else(|| {
        BusDayCalendar::new(
            weekmask_array,
            holidays.map_or(Array1::from_vec(vec![]), |h| {
                Array1::from_iter(h.data().iter().map(|x| x.to_i64().unwrap_or(0)))
            }),
        )
    });

    let mut result = Array::from_shape_vec(dates.shape().to_vec(), vec![false; dates.size()])?;

    for i in 0..dates.size() {
        let date = dates.get_linear(i).copied().unwrap_or(T::zero());
        let is_business_day = check_is_business_day(date, &calendar)?;
        result.set_linear(i, is_business_day);
    }

    Ok(result)
}

pub fn datetime_as_string<T>(
    arr: &Array<T>,
    unit: Option<&str>,
    timezone: &str,
    casting: &str,
) -> Result<Array<String>, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Zero + Copy + std::fmt::Debug + 'static,
{
    let unit_str = unit.unwrap_or("s");
    let timezone_offset = parse_timezone(timezone)?;

    if !matches!(casting, "safe" | "unsafe" | "same_kind") {
        return Err(NumPyError::value_error("Invalid casting mode", "datetime"));
    }

    let mut result = Array::from_shape_vec(arr.shape().to_vec(), vec![String::new(); arr.size()])?;

    for i in 0..arr.size() {
        let timestamp = arr.get_linear(i).copied().unwrap_or(T::zero());
        let string_value = format_datetime(timestamp, unit_str, timezone_offset)?;
        result.set_linear(i, string_value);
    }

    Ok(result)
}

fn parse_weekmask(weekmask: &str) -> Result<[bool; 7], NumPyError> {
    let mut mask = [false; 7];

    if weekmask == "1111100" {
        mask = [false, true, true, true, true, true, false];
    } else if weekmask == "1111111" {
        mask = [true, true, true, true, true, true, true];
    } else {
        let chars: Vec<char> = weekmask.chars().collect();
        if chars.len() != 7 {
            return Err(NumPyError::value_error(
                "Weekmask must have 7 characters",
                "datetime",
            ));
        }

        for (i, c) in chars.iter().enumerate() {
            mask[i] = *c == '1';
        }
    }

    Ok(mask)
}

fn parse_timezone(timezone: &str) -> Result<i64, NumPyError> {
    match timezone {
        "UTC" => Ok(0),
        "EST" => Ok(-18000),
        "CST" => Ok(-21600),
        "MST" => Ok(-25200),
        "PST" => Ok(-28800),
        _ => {
            if timezone.starts_with("UTC") {
                let offset_str = &timezone[3..];
                match offset_str.parse::<i64>() {
                    Ok(offset) => Ok(offset * 3600),
                    Err(_) => Err(NumPyError::value_error("Invalid UTC offset", "datetime")),
                }
            } else {
                Err(NumPyError::value_error("Unsupported timezone", "datetime"))
            }
        }
    }
}

fn count_business_days<T>(begin: T, end: T, calendar: &BusDayCalendar) -> Result<i64, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Copy + std::fmt::Debug,
{
    let begin_i64 = T::to_i64(&begin).unwrap_or(0);
    let end_i64 = T::to_i64(&end).unwrap_or(0);

    if begin_i64 >= end_i64 {
        return Ok(0);
    }

    let mut count = 0;
    let mut current = begin_i64;

    while current < end_i64 {
        if is_business_day_internal(current, calendar) {
            count += 1;
        }
        current += 86400;
    }

    Ok(count)
}

fn apply_business_day_offset<T>(
    date: T,
    offset: T,
    roll: &str,
    calendar: &BusDayCalendar,
) -> Result<T, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Copy + std::fmt::Debug,
{
    let date_i64 = T::to_i64(&date).unwrap_or(0);
    let offset_i64 = T::to_i64(&offset).unwrap_or(0);

    let mut current = date_i64;
    let mut days_added = 0;

    while days_added < offset_i64.abs() {
        current += if offset_i64 > 0 { 86400 } else { -86400 };

        if is_business_day_internal(current, calendar) {
            days_added += 1;
        }
    }

    if !is_business_day_internal(current, calendar) {
        current = roll_to_business_day(current, roll, calendar)?;
    }

    Ok(T::from_i64(current).unwrap_or(date))
}

fn check_is_business_day<T>(date: T, calendar: &BusDayCalendar) -> Result<bool, NumPyError>
where
    T: FromPrimitive + ToPrimitive + Copy + std::fmt::Debug,
{
    let date_i64 = T::to_i64(&date).unwrap_or(0);
    Ok(is_business_day_internal(date_i64, calendar))
}

fn is_business_day_internal(timestamp: i64, calendar: &BusDayCalendar) -> bool {
    let days_since_epoch = timestamp / 86400;
    let day_of_week = ((days_since_epoch + 4) % 7) as usize;

    if !calendar.weekmask[day_of_week] {
        return false;
    }

    for &holiday in calendar.holidays.iter() {
        if holiday == timestamp {
            return false;
        }
    }

    true
}

fn roll_to_business_day(
    timestamp: i64,
    roll: &str,
    calendar: &BusDayCalendar,
) -> Result<i64, NumPyError> {
    match roll {
        "raise" => Err(NumPyError::value_error(
            "Non-business day encountered with roll='raise'",
            "datetime",
        )),
        "nat" => Ok(i64::MIN),
        "forward" | "following" => {
            let mut current = timestamp + 86400;
            while !is_business_day_internal(current, calendar) {
                current += 86400;
            }
            Ok(current)
        }
        "backward" | "preceding" => {
            let mut current = timestamp - 86400;
            while !is_business_day_internal(current, calendar) {
                current -= 86400;
            }
            Ok(current)
        }
        "modifiedforward" | "modifiedfollowing" => {
            let mut current = timestamp + 86400;
            while !is_business_day_internal(current, calendar) {
                current += 86400;
            }
            let days_since_epoch = current / 86400;
            let day_of_week = ((days_since_epoch + 4) % 7) as usize;

            if day_of_week == 6 {
                current -= 86400 * 2;
            } else if day_of_week == 0 {
                current -= 86400;
            }

            Ok(current)
        }
        "modifiedbackward" | "modifiedpreceding" => {
            let mut current = timestamp - 86400;
            while !is_business_day_internal(current, calendar) {
                current -= 86400;
            }
            let days_since_epoch = current / 86400;
            let day_of_week = ((days_since_epoch + 4) % 7) as usize;

            if day_of_week == 6 {
                current += 86400;
            } else if day_of_week == 0 {
                current += 86400 * 2;
            }

            Ok(current)
        }
        _ => Err(NumPyError::value_error("Invalid roll option", "datetime")),
    }
}

fn format_datetime<T>(timestamp: T, unit: &str, timezone_offset: i64) -> Result<String, NumPyError>
where
    T: FromPrimitive + ToPrimitive + std::fmt::Debug,
{
    let base_timestamp = T::to_i64(&timestamp).unwrap_or(0);
    let multiplier = match unit {
        "Y" => 31536000,
        "M" => 2592000,
        "W" => 604800,
        "D" => 86400,
        "h" => 3600,
        "m" => 60,
        "s" => 1,
        "ms" => 1,
        "us" => 1,
        "ns" => 1,
        _ => return Err(NumPyError::value_error("Invalid datetime unit", "datetime")),
    };

    let seconds = base_timestamp * multiplier + timezone_offset;

    let datetime = DateTime::from_timestamp(seconds, 0);
    match datetime {
        Some(dt) => Ok(dt.format("%Y-%m-%d %H:%M:%S").to_string()),
        None => Ok("Invalid datetime".to_string()),
    }
}
