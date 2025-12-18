use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use anyhow::Result;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: inspect_onnx <path_to_model.onnx>");
        return Ok(());
    }

    let model_path = &args[1];
    println!("Inspecting model: {}", model_path);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    println!("\n--- Inputs ---");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("#{}: Name: {}", i, input.name);
        println!("    Type: {:?}", input.input_type);
    }

    println!("\n--- Outputs ---");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("#{}: Name: {}", i, output.name);
        println!("    Type: {:?}", output.output_type);
    }

    Ok(())
}
