"use client";

import { useEffect, useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sex, setSex] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [isFullscreen, setisFullscreen] = useState(false);

  useEffect (() => {
    const handleEsc = (e: KeyboardEvent) => {
      if(e.key === "Escape")
        setisFullscreen(false);
      };
      window.addEventListener("keydown", handleEsc);
      return () => window.removeEventListener("keydown", handleEsc);
    }, []);

  const handleFile = (selectedFile: File) => {
    if(!sex){
      alert("Please select the sex of the blood cockle before uploading an image.");
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (!sex) return;
    if (e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async () => {
    if(!sex){
      alert("Please select the sex");
      return;
    }
    
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

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="w-3/4 lg:w-2/3 mx-auto p-8shadow-2xl rounded-3xl">
      {isFullscreen && preview && (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/95 backdrop-blur-sm p-4 animate-in fade-in duration-300 cursor-zoom-out
        onclick={() => setisFullscreen(false)}
      ">
        <button className="absolute top-6 right-6 text-white/50 hover:text-white transition-colors"
            onClick={() => setisFullscreen(false)} >
          <p className="text-2xl">X</p>
        </button>
        <img src={preview} 
        alt="Fullscreen preview"
        className="max-w-full max-h-full object-contain rounded-lg shadow-2xl animate-in zoom-in-95 duration-300" 
        />
      </div>
      )}
      <h2 className="text-3xl font-black text-center mb-8 text-blue-900 italic">
        GonaX - Blood Cockle Classifier
      </h2>

      {/*SEX SELECTION */}
      {!sex ? (
        <div className="space-y-6 animate-in fade-in zoom-in duration-300">
          <p className="text-center text-gray-600 font-medium">Select specimen sex to begin:</p>
          <div className="grid grid-cols-2 gap-6">
            <button
              onClick={() => setSex("male")}
              className="aspect-square flex flex-col items-center justify-center border-4 border-gray-100 rounded-3xl hover:border-blue-500 hover:bg-blue-50 transition-all group"
            >
              <span className="text-5xl mb-2 group-hover:scale-110 transition-transform">♂️</span>
              <span className="font-bold text-xl text-gray-700">MALE</span>
            </button>

            <button
              onClick={() => setSex("female")}
              className="aspect-square flex flex-col items-center justify-center border-4 border-gray-100 rounded-3xl hover:border-pink-500 hover:bg-pink-50 transition-all group"
            >
              <span className="text-5xl mb-2 group-hover:scale-110 transition-transform">♀️</span>
              <span className="font-bold text-xl text-gray-700">FEMALE</span>
            </button>
          </div>
        </div>
      ) : (
        // UPLOAD
        <div className="animate-in slide-in-from-bottom-4 duration-500 mb-10">
          <div className="flex justify-between items-center mb-6">
            <p className="font-bold text-gray-500 uppercase tracking-widest text-sm">
              Selected: <span className={sex === 'male' ? 'text-blue-600' : 'text-pink-600'}>{sex}</span>
            </p>
            <button 
              onClick={() => {setSex(null); setFile(null); setPreview(null); setResult(null);}}
              className="font-bold text-s text-blue-500 bg-blue-600 hover:bg-blue-800 rounded-2xl px-3 py-1 text-white transition opacity-80 hover:opacity-70 shadow-xl shadow-blue-100"
            >
              Change Sex
            </button>
          </div>
          {!preview ? (
            <label
            htmlFor="fileInput"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="flex flex-col items-center justify-center border-2 border-dashed border-blue-300 p-10 text-center rounded-2xl cursor-pointer hover:bg-blue-50 transition-all bg-gray-50"
          >
            <p className="text-gray-700 font-bold">Drag & drop image</p>
            <div>
              <svg></svg>
            </div>
            <p className="text-sm text-gray-400">or click to browse</p>

            <input
              id="fileInput"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            />
          </label>
          ):(
          <div className="animate-in fade-in zoom-in duration-300">
            <div className="relative group" onClick={() => setisFullscreen(true )}>
              <p className="text-xs font-bold text-gray-400 mb-2 uppercase">Image Preview:</p>
              <img src={preview} alt="preview" className="rounded-2xl shadow-lg border-4 border-white w-full h-100 object-cover" />
              <button 
              onClick={clearImage}
              className="absolute top-8 right-4 bg-white/90 backdrop-blur-sm text-red-500 font-bold text-xs px-4 py-2 rounded-full shadow-md hover:bg-red-500 hover:text-white transition-all"
              >
                Remove Image
              </button>
            </div>
          </div>
          )}

          <button
            onClick={handleSubmit}
            disabled={loading || !file}
            className="mt-8 w-full bg-blue-600 text-white p-4 rounded-2xl font-bold text-lg hover:bg-blue-700 transition disabled:opacity-50 shadow-xl shadow-blue-100"
          >
            {loading ? "Processing XGBoost..." : "Analyze Specimen"}
          </button>
            {result && (
              <div className="mt-8 p-6 rounded-2xl bg-gradient-to-br from-blue-600 to-blue-800 text-white shadow-xl">
                <h3 className="font-bold text-sm uppercase tracking-[0.2em] opacity-80 mb-4">Classification Result</h3>
                <div className="flex justify-between items-end">
                  <div>
                    <p className="text-xs opacity-70 mb-1">Predicted Stage</p>
                    <p className="text-3xl font-black capitalize">{result.predicted_stage}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs opacity-70 mb-1">Confidence</p>
                    <p className="text-2xl font-black">{(result.confidence * 100).toFixed(2)}%</p>
                  </div>
                </div>
              </div>
            )}
          <br></br>
        </div>
      )}
    </div>
  );
}