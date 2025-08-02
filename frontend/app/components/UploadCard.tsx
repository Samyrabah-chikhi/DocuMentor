"use client";
import React, { ChangeEvent, useRef, useState } from "react";
import { FileText, FileCheck, Send } from "lucide-react";
import UploadCards from "./UploadCards";

type Output = "summary" | "answer";

export default function UploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const [buttonText, setButtonText] = useState<Output>("summary");
  const [question, setQuestion] = useState<String>("");

  const inputRef = useRef(null);

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  }

  const handleClickInput = () => {
    inputRef.current.click();
  };

  const handleGetSummary = () => {
    if(buttonText == "summary"){
      console.log("Get Summary")
    }
    else{
      console.log(`Ask question: ${question}`)
    }
  };

  const handleRadioChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setButtonText(value);
  };

  return (
    <div
      className="bg-white shadow-lg rounded-2xl w-[50vw]
       border-2 border-gray-200 
    flex flex-col p-6 gap-4"
    >
      <section className="flex gap-2 items-center">
        <FileText className="h-5 w-5" />
        <h2 className="text-lg text-black font-semibold">Upload PDF</h2>
      </section>
      <p className="text-sm">Drag and drop your PDF file or click to browse</p>
      <main
        className={`mt-4 border-2 border-dashed rounded-xl w-full h-full p-6  ${
          file
            ? "border-green-500 bg-green-50"
            : "border-gray-300 hover:border-gray-500 "
        }`}
      >
        {file ? (
          <div className="flex flex-col items-center justify-center">
            <FileCheck className="h-12 w-12 text-green-500"></FileCheck>
            <h1 className="text-lg font-semibold text-black">{file.name}</h1>
            <p>{(file.size / 1024).toFixed(2)} KB</p>
          </div>
        ) : (
          <UploadCards
            handleFileChange={handleFileChange}
            handleClickInput={handleClickInput}
            inputRef={inputRef}
          ></UploadCards>
        )}
      </main>
      {file && (
        <section className="flex flex-col gap-4 items-center justify-center w-full">
          <label
            className="flex items-center gap-3 w-full p-3 
          border border-gray-300 rounded-lg"
          >
            <input
              type="radio"
              name="output"
              value="summary"
              checked={buttonText == "summary"}
              onChange={handleRadioChange}
            />
            <h2 className="text-black font-semibold">Summarize PDF</h2>
            <p className="font-medium">Get a concise summary of the document</p>
          </label>

          <label
            className="flex items-center gap-3 w-full p-3 
          border border-gray-300 rounded-lg"
          >
            <input
              type="radio"
              name="output"
              value="answer"
              checked={buttonText == "answer"}
              onChange={handleRadioChange}
            />
            <h2 className="text-black font-semibold">Ask a Question</h2>
            <p className="font-medium">
              Get specific answers from the document
            </p>
          </label>

          {buttonText == "answer" && (
            <div className="flex flex-col gap-2 w-full">
              <h2 className="text-black font-medium">Your Question</h2>
              <textarea
                name="question"
                placeholder="What would you like to know about this PDF?"
                className="p-2 border border-gray-300 rounded-lg focus:outline-none 
                focus:ring-3 focus:ring-stone-600 duration-300 text-black"
                rows={3}
                value={question}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                  setQuestion(e.target.value);
                }}
              />
            </div>
          )}

          <button
            className={`text-white font-semibold rounded-lg w-full p-3
            flex items-center justify-center gap-4
           bg-neutral-900 hover:bg-neutral-700 duration-200 cursor-pointer
           disabled:bg-stone-400 disabled:cursor-auto`}
            onClick={handleGetSummary}
            disabled={buttonText == "answer" && question == "" }
          >
            <Send></Send>
            <p>{buttonText == "summary" ? "Get Summary" : "Get Answer"}</p>
          </button>
        </section>
      )}
    </div>
  );
}
