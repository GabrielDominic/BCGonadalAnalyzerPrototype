"use client";

import { useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sex, setSex] = useState("male");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload an image");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sex", sex);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error processing image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-xl rounded-2xl"> <h2 className="text-2xl font-bold text-center mb-6"> GonaX - Blood Cockle Classifier </h2>

      {/* Upload Area */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput")?.click()}
        className="border-2 border-dashed border-blue-400 p-8 text-center rounded-xl cursor-pointer hover:bg-blue-50 transition"
      >
        <p className="text-gray-700 font-medium">
          Drag & drop an image here
        </p>
        <p className="text-sm text-gray-500 mt-1">
          or click to upload
        </p>

        <input
          id="fileInput"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) =>
            e.target.files && handleFile(e.target.files[0])
          }
        />
      </div>
      {/* Preview */}
      {preview && (
        <img
          src={preview}
          alt="preview"
          className="mt-4 rounded-xl shadow"
        />
      )}

      {/* Sex Selection */}
      <div className="mt-4">
        <p className="font-semibold mb-2">Select Sex:</p>

        <div className="flex gap-4">
          <label className="flex items-center gap-2">
            <input
              type="radio"
              value="male"
              checked={sex === "male"}
              onChange={(e) => setSex(e.target.value)}
            />
            Male
          </label>

          <label className="flex items-center gap-2">
            <input
              type="radio"
              value="female"
              checked={sex === "female"}
              onChange={(e) => setSex(e.target.value)}
            />
            Female
          </label>
        </div>
      </div>

      {/* Button */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="mt-6 w-full bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 transition disabled:opacity-50"
      >
        {loading ? "Processing..." : "Analyze"}
      </button>

      {/* RESULT DISPLAY */}
      {result && (
        <div className="mt-6 p-4 border rounded-xl bg-gray-50">
          <h3 className="font-bold text-lg mb-2">Result</h3>

          <p>
            <span className="font-semibold">Stage:</span>{" "}
            {result.predicted_stage}
          </p>

          <p>
            <span className="font-semibold">Confidence:</span>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
);
}