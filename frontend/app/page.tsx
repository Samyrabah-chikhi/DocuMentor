import UploadCard from "./components/UploadCard";

export default function Home() {
  return (
    <div className="flex flex-col justify-between items-center gap-2 p-10 ">
      <h1 className="text-3xl text-black font-semibold">PDF Assistant</h1>
      <p className="text-md mb-4">
        Upload a PDF to get summaries or ask questions about its content
      </p>
      <UploadCard></UploadCard>
    </div>
  );
}
