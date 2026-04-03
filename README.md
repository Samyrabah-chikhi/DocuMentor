# DocuMentor

![Next.js](https://img.shields.io/badge/Next.js-000?logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-38B2AC?logo=tailwindcss&logoColor=white)

---

## 📝 Description

DocuMentor is a web application for analyzing PDF documents using local Large Language Models (LLMs).  
It allows users to upload documents, generate summaries, and extract key information while keeping all data processed locally.

---

## ✨ Features

- 📄 Upload and analyze PDF files  
- 🧠 Local LLM-based summarization  
- 🔍 Key information extraction  
- ⚡ Fast and minimal web interface  
- 🔒 Local processing (no external APIs)

---

## 🛠️ Tech Stack

- **Frontend:** Next.js, React, TypeScript  
- **Styling:** Tailwind CSS  
- **Backend / Processing:** Python (LLM + PDF processing)

---

## 📦 Dependencies

### Frontend

- next
- react
- react-dom
- typescript
- tailwindcss
- postcss
- lucide-react

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd DocuMentor
```
### 2. Install frontend dependencies
```bash
cd frontend
npm install
```
### 3. Run the backend (Python)
```bash
python BookSummary.py
```
### 4. Run the frontend
```bash
npm run dev
```
--- 

## 📁 Project Structure
```
.
├── BookSummary.py        # PDF processing + LLM logic
├── context/              # Input documents
└── frontend/
    ├── app/
    │   ├── components/   # UI components
    │   ├── layout.tsx
    │   └── page.tsx
    ├── public/
    └── package.json
```
---

## 🧪 Usage
- Run the Python script
- Start the frontend
- Upload a PDF
- View generated summary and extracted data

## 📌 Notes
- Uses local models for processing
- No external API calls
- Designed for fast document analysis
