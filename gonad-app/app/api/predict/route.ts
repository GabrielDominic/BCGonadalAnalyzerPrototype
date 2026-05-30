import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try{

    const formData = await req.formData();
    const BACKEND_URL = process.env.BACKEND_URL || "http://http://127.0.0.1:8000/predict";
  
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    });
  
    if(!res.ok) {
      const errorText = await res.json();
      console.error("Python Server Error:", errorText);
      return NextResponse.json({ error: "Error processing image" }, { status: res.status  }        
      );
    }
  
    const data = await res.json();
    return NextResponse.json(data);
  
    } catch (error: any) {
      console.error("Unexpected Error:", error);
      return NextResponse.json({ error: "Unexpected error occurred" }, { status: 500 });
    }
  }