"use client";
import React, {
  ChangeEvent,
  Dispatch,
  SetStateAction,
  useRef,
  useState,
} from "react";
import { FileText, FileCheck, Send } from "lucide-react";
import UploadCards from "./UploadCards";

type Output = "summary" | "answer";

const RadioOption = ({
  value,
  selected,
  onChange,
  title,
  description,
}: {
  value: Output;
  selected: Output;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  title: string;
  description: string;
}) => (
  <label className="flex items-center gap-3 w-full p-3 border border-gray-300 rounded-lg">
    <input
      type="radio"
      name="output"
      value={value}
      checked={selected === value}
      onChange={onChange}
    />
    <div>
      <h2 className="text-black font-semibold">{title}</h2>
      <p className="font-medium">{description}</p>
    </div>
  </label>
);

export default function UploadCard({
  setAnswer,
  answer,
}: {
  setAnswer: Dispatch<SetStateAction<String>>;
  answer: String;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [selectedOutput, setSelectedOutput] = useState<Output>("summary");
  const [question, setQuestion] = useState<string>("");

  const [loading, setLoading] = useState<Boolean>(false);

  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleClickInput = () => inputRef.current?.click();

  const handleGetSummary = async () => {
    const formData = new FormData();
    formData.append("file", file);

    if (selectedOutput === "summary") {
      setLoading(true);
      try {
        const res = await fetch("http://127.0.0.1:8000", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setAnswer(data.Summary);
        console.log("Summary Response:", data);
      } catch (err) {
        console.error("Error fetching summary:", err);
      }
    } else {
      console.log(`Ask question: ${question}`);
    }
    setLoading(false);
  };

  const handleRadioChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSelectedOutput(e.target.value as Output);
  };

  const isDisabled =
    loading || (selectedOutput === "answer" && question.trim() === "");

  return (
    <div
      className={`bg-white shadow-lg rounded-2xl w-[45vw]
      ${file ? "h-[100vh]" : "h-[60vh]"}
      ${file && selectedOutput === "answer" ? "h-[125vh]" : ""}
    border-2 flex flex-col p-6 gap-4`}
    >
      <section className="flex gap-2 items-center">
        <FileText className="h-5 w-5" />
        <h2 className="text-lg text-black font-semibold">Upload PDF</h2>
      </section>

      <p className="text-sm">Drag and drop your PDF file or click to browse</p>

      <main
        className={`mt-4 border-2 border-dashed rounded-xl w-full p-6 relative ${
          file
            ? "border-green-500 bg-green-50 h-auto"
            : "border-gray-300 hover:border-gray-500 h-40"
        }`}
      >
        <button
          className="absolute top-1 right-1 rounded-lg cursor-pointer
         px-4 py-0.5 bg-red-500 text-white"
          onClick={() => {
            setFile(null);
          }}
        >
          X
        </button>
        {file ? (
          <div className="flex flex-col items-center justify-center">
            <FileCheck className="h-12 w-12 text-green-500" />
            <h1 className="text-lg font-semibold text-black">{file.name}</h1>
            <p>{(file.size / 1024).toFixed(2)} KB</p>
          </div>
        ) : (
          <UploadCards
            handleFileChange={handleFileChange}
            handleClickInput={handleClickInput}
            inputRef={inputRef}
          />
        )}
      </main>

      {file && (
        <section className="flex flex-col gap-4 items-center justify-center w-full">
          <RadioOption
            value="summary"
            selected={selectedOutput}
            onChange={handleRadioChange}
            title="Summarize PDF"
            description="Get a concise summary of the document"
          />

          <RadioOption
            value="answer"
            selected={selectedOutput}
            onChange={handleRadioChange}
            title="Ask a Question"
            description="Get specific answers from the document"
          />

          {selectedOutput === "answer" && (
            <div className="flex flex-col gap-2 w-full">
              <h2 className="text-black font-medium">Your Question:</h2>
              <textarea
                name="question"
                placeholder="What would you like to know about this PDF?"
                className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-stone-500 duration-300 text-black"
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
            </div>
          )}

          <button
            className={`text-white font-semibold rounded-lg w-full p-3 flex items-center justify-center gap-4 bg-neutral-900 hover:bg-neutral-700 duration-200 cursor-pointer disabled:bg-stone-400 disabled:cursor-not-allowed`}
            onClick={handleGetSummary}
            disabled={isDisabled}
          >
            <Send />
            <p>{selectedOutput === "summary" ? "Get Summary" : "Get Answer"}</p>
          </button>
        </section>
      )}
      
    </div>
  );
}
