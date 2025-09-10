from transformers import AutoModel, AutoTokenizer

if False:
    #model_name = "meta-llama/Llama-3.1-8B-Instruct"
    #save_path = "./Llama-3.1-8B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    save_path = "./models/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

if True:
    import whisper
    model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
