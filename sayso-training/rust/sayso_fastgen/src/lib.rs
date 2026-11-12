pub mod yaml;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

/// dump_entities(data: list[dict[str, str | None]]) -> str | None
///
/// Same text as the generator's PyYAML `_dump`, or None when the input uses something this
/// emitter does not cover (caller falls back to PyYAML).
#[pyfunction]
fn dump_entities(data: &Bound<'_, PyList>) -> PyResult<Option<String>> {
    // Borrow the UTF-8 buffers Python already holds instead of copying every name.
    let mut held: Vec<Vec<(Bound<'_, PyString>, Option<Bound<'_, PyString>>)>> = Vec::with_capacity(data.len());
    for item in data.iter() {
        let Ok(dict) = item.downcast_into::<PyDict>() else { return Ok(None) };
        let mut entry = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let Ok(key) = k.downcast_into::<PyString>() else { return Ok(None) };
            let value = if v.is_none() {
                None
            } else if let Ok(s) = v.downcast_into::<PyString>() {
                Some(s)
            } else {
                return Ok(None);
            };
            entry.push((key, value));
        }
        held.push(entry);
    }
    let mut entries: Vec<Vec<(&str, Option<&str>)>> = Vec::with_capacity(held.len());
    for entry in &held {
        let mut row = Vec::with_capacity(entry.len());
        for (k, v) in entry {
            row.push((k.to_str()?, match v { Some(s) => Some(s.to_str()?), None => None }));
        }
        entries.push(row);
    }
    Ok(yaml::dump_entries(&entries))
}

#[pymodule]
fn sayso_fastgen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dump_entities, m)?)?;
    Ok(())
}
