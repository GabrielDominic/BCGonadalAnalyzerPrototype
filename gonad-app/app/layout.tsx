// import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gradient-to-br from-blue-50 to-gray-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}