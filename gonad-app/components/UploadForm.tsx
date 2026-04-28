"use client";

import { useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sex, setSex] = useState("male");

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
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

    const formData = new FormData();
    formData.append("image", file);
    formData.append("sex", sex);

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    console.log(data);
  };

  return (
    <div className="max-w-md mx-auto p-6 border rounded-xl shadow">
      <h2 className="text-xl font-bold mb-4">
        Gonadal Stage Classifier
      </h2>

      {/* Drag & Drop Area */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="border-2 border-dashed p-6 text-center rounded-lg cursor-pointer"
      >
        <p>Drag & drop an image here</p>
        <p className="text-sm text-gray-500">or click below</p>

        <input
          type="file"
          accept="image/*"
          className="mt-3"
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
          className="mt-4 rounded-lg"
        />
      )}

      {/* Radio Buttons */}
      <div className="mt-4">
        <p className="font-semibold mb-2">Select Sex:</p>

        <label className="mr-4">
          <input
            type="radio"
            value="male"
            checked={sex === "male"}
            onChange={(e) => setSex(e.target.value)}
          />{" "}
          Male
        </label>

        <label>
          <input
            type="radio"
            value="female"
            checked={sex === "female"}
            onChange={(e) => setSex(e.target.value)}
          />{" "}
          Female
        </label>
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="mt-6 w-full bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700"
      >
        Submit
      </button>
    </div>
  );
}