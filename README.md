# localCowork — Your Local AI Assistant 

**localCowork** is an AI-powered assistant (Inspired by Cluade Cowork) that lives on your computer. It takes your natural-language requests and turns them into actions, helping you organize your digital life without you having to lift a finger.

Think of it as a smart middleman between you and your files. You tell it what you want in plain English, and it figures out the steps to make it happen.

---

## ✨ What can it do for you?

- **Organize Your Files**: "Find all images in my downloads from today and move them to a new folder called 'Daily Photos'."
- **Find Information**: "Check if I have any resumes in my downloads and tell me what's in them."
- **Manage Documents**: "List all my PDF files and tell me which ones are larger than 5MB."
- **Safe & Secure**: Everything runs locally on your machine. Your files never leave your computer, and any complex logic is handled in a secure, isolated "sandbox" for your protection.

---

## 🚀 How to use it

### 1. Installation

Getting started is easy. Just clone the project and install it:

```bash
git clone https://github.com/yourusername/localCowork.git
cd localCowork
uv pip install -e .
```

### 2. Running Tasks

Once installed, you can talk to localCowork directly from your terminal:

```bash
localcowork run "Organize my downloads folder by grouping files into 'Images', 'PDFs', and 'Others'."
```

### 3. What Happens Next?

- **The Plan**: localCowork explains how it's going to achieve your goal.
- **The Work**: It performs the file moves, searches, and analysis for you.
- **The Result**: You get a friendly summary of exactly what was accomplished.

---

## 🤝 Why localCowork?

- **Privacy First**: Your data stays with you. No cloud processing of your personal files.
- **Human-Friendly**: No complex commands to learn. If you can type it, localCowork can try to do it.
- **Transparent**: You see exactly what the AI is planning to do before it does it.
