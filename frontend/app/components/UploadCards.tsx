import React from "react";
import { Upload} from "lucide-react";

export default function UploadCards({
  inputRef,
  handleFileChange,
  handleClickInput,
}: {
  inputRef:React.RefObject<null>;
  handleFileChange:(e: React.ChangeEvent<HTMLInputElement>) => void;
  handleClickInput:() => void;
}) {
  return (
    <div>
      <span className="mt-auto flex flex-col w-full items-center gap-2">
        <Upload className="h-12 w-12 text-gray-400"></Upload>
        <p className="text-md text-gray-700">Drop your PDF here, or</p>
      </span>
      <div className="flex w-fit">
        <input
          type="file"
          ref={inputRef}
          onChange={handleFileChange}
          accept=".pdf"
          className="hidden"
        ></input>
        <p
          className="text-blue-600 hover:text-blue-800 cursor-pointer font-medium"
          onClick={handleClickInput}
        >
          Browse files
        </p>
      </div>
    </div>
  );
}
