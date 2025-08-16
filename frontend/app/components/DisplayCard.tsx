import React from "react";

export default function DisplayCard({ answer }: { answer: String }) {
  return (
    <div
      className="bg-white shadow-lg rounded-2xl w-[45vw] h-[120vh]
    border-2 flex flex-col p-6 gap-4 overflow-y-auto"
    >
      {answer != "" && (
        <div>
          {answer.split("\n").map((line, index) => {
            if (line.startsWith("**") && line.endsWith("**")) {
              return (
                <header key={index}>
                  <br />
                  <h2 className="text-xl text-black font-medium mb-2">
                    {line.replace(/\*\*/g, "")}
                  </h2>
                </header>
              );
            }
            return (
              <p className="indent-8" key={index}>
                {line}
              </p>
            );
          })}
        </div>
      )}
    </div>
  );
}
