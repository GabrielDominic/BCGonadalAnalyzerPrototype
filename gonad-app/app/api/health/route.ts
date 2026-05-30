import { NextResponse } from "next/server";

export async function GET() {
  const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json({ status: "backend offline" }, { status: 500 });
  }
}