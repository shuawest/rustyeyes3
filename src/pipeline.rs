use image::{ImageBuffer, Rgb};
use anyhow::Result;
use crate::types::PipelineOutput;

pub trait Pipeline {
    fn name(&self) ->String;
    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>>;
    // We could move drawing to the OutputSink, but for now the pipeline knows best how to visualize its output?
    // Or simpler: Pipeline returns Output, Main/Output module draws it. 
    // Let's stick to returning Output. The Output module (window) will need to know how to draw PipelineOutput.
}
