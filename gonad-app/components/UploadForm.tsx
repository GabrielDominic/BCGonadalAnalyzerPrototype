"use client";

import { useEffect, useState } from "react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sex, setSex] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [isFullscreen, setisFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelChoice, setModelChoice] = useState<"ml" | "dl">("ml");

  useEffect (() => {
    // Wake up Backend
    fetch("/api/health").catch(() => {});

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
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sex", sex);
    formData.append("model_choice", modelChoice);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const contentType = res.headers.get("content-Type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Server returned a non-JSON response. Is the API route correct?");
      }
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || "Error processing image");
        // alert(data.detail || "An error occurred while processing the image.");
        setLoading(false);
        return;
      }

      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError( err.message || "An error occurred while processing the image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const getModelDescription = () => {
    if (modelChoice === "ml") {
      if (sex === "male") {
        return(
          <>
          <span className="font-bold text-blue-800">Best for:</span> Detecting
          <span className="font-semibold text-blue-700"> Mature</span> and 
          <span className="font-semibold text-blue-700"> Spent</span> stages.
          <br />Better at detecting <span className="font-bold text-blue-800">Developing stage</span> compared to DL model.
          </>
        );
      } else {
        return(
          <>
          <span className="font-bold text-blue-800">Best for:</span> Detecting
          <span className="font-semibold text-blue-700"> Mature</span> stage
          <br />Decent performance on
          <span className="font-semibold text-blue-700"> Spawning</span> and 
          <span className="font-semibold text-blue-700"> Spent</span> stages.
          </>
        );
      }
    } else {
      if (sex === "male") {
        return(
          <>
          <span className="font-bold text-blue-800">Best for:</span> Detecting
          <span className="font-semibold text-blue-700"> Mature</span> and 
          <span className="font-semibold text-blue-700"> Spawning</span> stages.
          </>
        );
      } else {
        return(
          <>
          <span className="font-bold text-blue-800">Best for:</span> Detecting
          <span className="font-semibold text-blue-700"> Developing</span> and 
          <span className="font-semibold text-blue-700"> Mature</span> stages.
          <br />Performs better on <span className="font-bold text-blue-800">Mature stage</span> compared to ML model.
          </>
        );
      }
    }
  };

  return (
    <div className="w-3/4 lg:w-2/3 mx-auto p-8shadow-2xl rounded-3xl">
      {isFullscreen && preview && (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/95 backdrop-blur-sm p-4 animate-in fade-in duration-300 cursor-zoom-out"
        onClick={() => setisFullscreen(false)}
      >
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
              onClick={() => {setSex(null); clearImage(); setError(null);}}
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
              onClick={(e) => {e.stopPropagation(); clearImage(); setError(null); }}
              className="absolute top-8 right-4 bg-white/90 backdrop-blur-sm text-red-500 font-bold text-xs px-4 py-2 rounded-full shadow-md hover:bg-red-500 hover:text-white transition-all"
              >
                Remove Image
              </button>
            </div>
          </div>
          )}
          {/* BOUNCER ALERT */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border-1-4 border-red-500 rounded-2xl shadow-md animate-in fade-in duration-500">
              <div className="flex items-center gap-3">
                  <span className="text-red-500 text-xl font-bold">⚠️</span>
                  <div>
                    <p className="text-red-800 font-bold text-sm uppercase tracking-tight">Anomaly Detected</p>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
              </div>
            </div>
          )}
          {/* Model Selection */}
          <div className="mb-6 animate-in fade-in duration-500">
            <p className="text-xs font-bold text-gray-400 mb-3 uppercase tracking-widest">Select Classifier Engine:</p>
            <div className="flex p-1 bg-gray-100 rounded-2xl border border-gray-200">
              <button
                onClick={() => setModelChoice("ml")}
                className={`flex-1 py-2 px-4 rounded-xl text-sm font-bold transition-all ${
                  modelChoice === "ml" 
                  ? "bg-white text-blue-600 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {sex === "male" ? "ML (XGBOOST)" : "ML (XGBOOST)"}
              </button>
              <button
                onClick={() => setModelChoice("dl")}
                className={`flex-1 py-2 px-4 rounded-xl text-sm font-bold transition-all ${
                  modelChoice === "dl"
                  ? "bg-white text-blue-600 shadow-sm" 
                  : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {sex === "male" ? "DL (EFFNET-B0)" : "DL (RESNET-50)"}
              </button>
            </div>
            {/* Model Strength */}
            <div className="mt-3 p-3 bg-blue-50/50 border border-blue-100 rounded-xl flex items-start gap-2 animate-in slide-in-from-top-1 duration-300">
              <span className="text-blue-500 mt-0.5">💡</span>
              <p className="text-[11px] leading-relaxed text-blue-600 italic">
                {getModelDescription()}
              </p>
            </div>
          </div>
          {/* Analyze */}
          <button
            onClick={handleSubmit}
            disabled={loading || !file}
            className="mt-8 w-full bg-blue-600 text-white p-4 rounded-2xl font-bold text-lg hover:bg-blue-700 transition disabled:opacity-50 shadow-xl shadow-blue-100"
          >
            {loading ? (
              <div className="flex items-center justify-center space-x-4">
                  <p>Processing Specimen...</p>
                  <div className="w-6 h-6 border-4 border-blue-500 rounded-full animate-spin border-t-transparent"></div>
              </div>
            ) : ("Analyze Specimen")}
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
                  <div className="mt-6 pt-6 border-t border-white/20">
                    <p className="text-xs opacity-70 mb-3 uppercase tracking-widest font-bold">
                      Stage Probability Breakdown:
                    </p>
                    <div className="space-y-3">
                      {Object.entries(result.probabilities).map(([stage, score]) => (
                        <div key={stage} className="group">
                          <div className="flex justify-between text-sm mb-1 capitalize">
                            <span className={stage === result.predicted_stage ? "font-bold" : "opacity-80"}>
                              {stage}
                            </span>
                            <span className="font-mono">{(Number(score) * 100).toFixed(1)}%</span>
                          </div>
                          {/* Progress Bar Background */}
                          <div className="w-full bg-white/20 h-1.5 rounded-full overflow-hidden">
                            {/* Progress Bar Fill */}
                            <div 
                              className={`h-full transition-all duration-1000 ease-out ${
                                stage === result.predicted_stage ? "bg-white" : "bg-white/40"
                              }`}
                              style={{ width: `${Number(score) * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
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