import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();

  const res = await fetch("http://127.0.0.1:8000/", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();

  return NextResponse.json(data);
}