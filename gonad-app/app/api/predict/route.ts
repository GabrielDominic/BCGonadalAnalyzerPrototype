import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try{

    const formData = await req.formData();
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    console.log(`Forwarding request to: ${BACKEND_URL}/predict`);
  
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const responseText = await res.text();
  
    if(!res.ok) {
      const errorText = JSON.parse(responseText);
      console.error("Python Server Error:", errorText);
      return NextResponse.json({ error: "Error processing image" }, { status: res.status  }        
      );
    }
  
    const data = JSON.parse(responseText);
    return NextResponse.json(data);
  
    } catch (error: any) {
      console.error("Unexpected Error:", error);
      return NextResponse.json({ error: "Unexpected error occurred" }, { status: 500 });
    }
  }