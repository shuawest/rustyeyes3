use anyhow::{Error, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, Module};
use candle_transformers::models::moondream::{Config, Model};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== Inspecting Moondream Loading ===");
    
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    println!("Initializing HF API...");
    let api = Api::new()?;
    let repo_v1 = api.repo(Repo::new("vikhyatk/moondream1".to_string(), RepoType::Model));
    let repo_v2 = api.repo(Repo::new("vikhyatk/moondream2".to_string(), RepoType::Model));
    
    println!("Fetching moondream1 model.safetensors...");
    let model_file = repo_v1.get("model.safetensors")?;
    println!("Model path: {:?}", model_file);
    
    println!("Fetching tokenizer.json (from moondream2)...");
    let tokenizer_file = repo_v2.get("tokenizer.json")?;
    println!("Tokenizer path: {:?}", tokenizer_file);
    
    // println!("Fetching config.json (from moondream1)...");
    // let config_file = repo_v1.get("config.json")?;
    // let mut config_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_file)?)?;

    println!("Using Hardcoded Config for Moondream1 Compatible Loading...");
    let mut config_json = serde_json::json!({
        "phi_config": {
            "vocab_size": 51200,
            "n_positions": 2048,
            "n_embd": 2048,
            "n_layer": 24,
            "n_head": 32,
            "n_inner": 8192,
            "rotary_dim": 32,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "tie_word_embeddings": false,
            "pad_vocab_size_multiple": 1,
            "use_flash_attn": false
        },
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 8192, // Hoping this controls Projection
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "image_size": 378,
            "patch_size": 14,
            "image_embedding_dim": 1152,
            "model_dim": 2048, 
            "hidden_dim": 1152,
            "hidden_features": 4304, // Matches Blocks [4304, 1152]
            "embed_len": 729,
            "embed_dim": 1152,
            "num_blocks": 27,
            "num_heads": 16,
            "act": "gelu",
            "projection_dim": 2048
        }
    });


    println!("Reparsing patched config...");
    let config: Config = serde_json::from_value(config_json)?;
    println!("Config parsed successfully.");

    println!("Parsing tokenizer...");
    let _tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(Error::msg)?;
    println!("Tokenizer parsed successfully.");
    
    println!("Loading Model Weights (may take time)...");
    
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(&model_file)? };
    let names = tensors.tensors();
    
    println!("--- Searching for Keys ---");
    for (name, _) in &names {
        if name.contains("patch") {
            println!("Found candidate: {}", name);
        }
    }
    println!("------------------------------------");
    
    let mut tensor_map = std::collections::HashMap::new();
    for (name, _) in names {
        let tensor = tensors.load(&name, &device)?;
        
        let mut new_name = if name.starts_with("model.text.blocks.") {
            name.replace("model.text.blocks.", "text_model.transformer.h.")
        } else if name == "model.text.wte" {
            "text_model.transformer.embd.wte.weight".to_string()
        } else if name.starts_with("model.text.post_ln.") {
            name.replace("model.text.post_ln.", "text_model.lm_head.ln.")
        } else if name.starts_with("model.text.lm_head.") {
            name.replace("model.text.lm_head.", "text_model.lm_head.linear.")
        } else if name.starts_with("model.vision.") {
            name.replace("model.vision.", "vision_encoder.encoder.model.visual.")
        } else {
            name.to_string()
        };

        if new_name.contains("patch_emb") && !new_name.contains("patch_embed") {
            new_name = new_name.replace("patch_emb", "patch_embed.linear");
        }
        if new_name.contains("vision") && new_name.contains("post_ln") {
             // Commonly 'post_ln' in ViT maps to 'norm' in Candle/HF
            new_name = new_name.replace("post_ln", "norm");
        }
        if new_name.contains("text_model") && new_name.contains("attn.qkv") {
            new_name = new_name.replace("attn.qkv", "mixer.Wqkv");
        }
        if new_name.contains("vision") && new_name.contains("proj_mlp") {
            new_name = new_name.replace("vision_encoder.encoder.model.visual.proj_mlp", "vision_encoder.projection.mlp");
        }
        if new_name.contains("text_model") && new_name.contains("attn.proj") {
            new_name = new_name.replace("attn.proj", "mixer.out_proj");
        }
        if new_name.contains("vision_encoder") {
            if new_name.contains("ln1") {
                new_name = new_name.replace("ln1", "norm1");
            }
            if new_name.contains("ln2") {
                new_name = new_name.replace("ln2", "norm2");
            }
        }
        if new_name.contains("pos_emb") && !new_name.contains("pos_embed") {
            new_name = new_name.replace("pos_emb", "pos_embed");
        }
        
        if new_name.contains("vision_encoder") && new_name.contains("0") && new_name.contains("attn") {
             println!("Mapped Vision Key: {}", new_name);
        }
        
        tensor_map.insert(new_name, tensor);
    }

    let vb = VarBuilder::from_tensors(tensor_map, DType::F32, &device);
    let _model = Model::new(&config, vb)?;
    println!("Model loaded successfully!");

    Ok(())
}
