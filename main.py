import string
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained("thevan2404/coteT5-phase2-v2")
tokenizer = AutoTokenizer.from_pretrained("thevan2404/coteT5-phase2-v2")
model.to(DEVICE)

def get_result(prefix, input_text):
    input_ids = tokenizer(str(prefix) + ": " + str(input_text), return_tensors="pt", max_length=512,
                          padding="max_length", truncation=True)

    summary_text_ids = model.generate(
        input_ids=input_ids["input_ids"].to(DEVICE),
        attention_mask=input_ids["attention_mask"].to(DEVICE),
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.2,
        top_k=5,
        top_p=0.95,
        max_length=48,
        min_length=2,
        num_beams=3,
    )
    result = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    if result[-1] in string.punctuation:
        result = result[:-1] + " " + result[-1]
    return result
import tkinter as tk
from tkinter import scrolledtext
import tkinter.font as tkFont

root = tk.Tk()
root.title("StackOverflow Title Generator")
root.configure(bg="#f0f4f8")

# Danh sách ngôn ngữ lập trình
languages = ["C#", "Java", "Python", "JS"]
selected_language = tk.StringVar()
selected_language.set(languages[0])  # Ngôn ngữ mặc định



# Hàm xử lý khi bấm nút "Generate Title"
def on_generate_title():
    desc = description_input.get("1.0", tk.END).strip()
    code = code_input.get("1.0", tk.END).strip()
    combined_input = f"<body> {desc} <code> {code}"
    language = selected_language.get()
    title = get_result(language, combined_input)

    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, title)
    output_text.config(state="disabled")

# ---- Giao diện ---- #
main_frame = tk.Frame(root, bg="#f0f4f8")
main_frame.pack(padx=20, pady=20)



# Ngôn ngữ
lang_frame = tk.Frame(main_frame, bg="#f0f4f8")
lang_frame.pack(fill="x", pady=(0, 15))
tk.Label(lang_frame, text="Programming Language:", bg="#f0f4f8", font=("Arial", 11)).pack(anchor="w")
tk.OptionMenu(lang_frame, selected_language, *languages).pack(anchor="w", pady=3)

# Description
tk.Label(main_frame, text="Description:", bg="#f0f4f8", font=("Arial", 11)).pack(anchor="w")
description_input = scrolledtext.ScrolledText(main_frame, height=5, width=70)
description_input.pack(pady=5)

# Code Snippet
tk.Label(main_frame, text="Code Snippet:", bg="#f0f4f8", font=("Arial", 11)).pack(anchor="w")
code_input = scrolledtext.ScrolledText(main_frame, height=5, width=70)
code_input.pack(pady=5)

# Nút sinh tiêu đề
generate_button = tk.Button(main_frame, text="Generate Title", command=on_generate_title,
                            bg="#4caf50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
generate_button.pack(pady=10)

# Kết quả
tk.Label(main_frame, text="Generated Title:", bg="#f0f4f8", font=("Arial", 11)).pack(anchor="w")
output_font = tkFont.Font(family="Arial", size=14, weight="bold")
output_text = scrolledtext.ScrolledText(main_frame, height=2, width=70, font=output_font, wrap="word", bg="#eef4fa")
output_text.pack(pady=10)
output_text.config(state="disabled")

root.mainloop()
