"use client";
import React, { ChangeEvent, useRef, useState } from "react";
import { FileText } from "lucide-react";
import UploadCards from "./UploadCards";

export default function UploadCard() {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef(null);

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  }

  const handleFileUpload = () => {
    inputRef.current.click();
  };

  return (
    <div
      className="bg-white shadow-lg rounded-2xl h-[50vh] w-[50vw]
       border-2 border-gray-200 
    flex flex-col p-6"
    >
      <section className="flex gap-2 items-center">
        <FileText className="h-5 w-5" />
        <h2 className="text-lg text-black font-semibold">Upload PDF</h2>
      </section>
      <p className="text-sm">Drag and drop your PDF file or click to browse</p>
      <main
        className={`mt-4 border-2 border-dashed rounded-xl w-full h-full p-6  ${
          file ? "border-green-500" : "border-gray-300 hover:border-gray-500 "
        }`}
      >
        {file ? (
          <div>
            <p>File name: {file.name}</p>
            <p>Size: {(file.size / 1024).toFixed(2)} KB</p>
            <p>Type: {file.type}</p>
          </div>
        ) : (
          <UploadCards
            handleFileChange={handleFileChange}
            handleFileUpload={handleFileUpload}
            inputRef={inputRef}
          ></UploadCards>
        )}
      </main>
    </div>
  );
}
