import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Flow Studio",
  description: "一个可交互的 RAG 检索流程实验室，用于观察文本分块、向量检索、BM25、混合融合、重排和答案生成。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
