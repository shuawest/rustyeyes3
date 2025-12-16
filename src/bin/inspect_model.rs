use ort::session::Session;
use anyhow::Result;

fn main() -> Result<()> {
    let model = Session::builder()?
        .commit_from_file("gaze_estimation_adas_0002.onnx")?;

    println!("Inputs:");
    for (i, input) in model.inputs.iter().enumerate() {
        println!("#{}: {} ({:?})", i, input.name, input.input_type);
    }

    println!("Outputs:");
    for (i, output) in model.outputs.iter().enumerate() {
        println!("#{}: {} ({:?})", i, output.name, output.output_type);
    }
    
    Ok(())
}
