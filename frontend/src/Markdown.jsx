import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

// Renders assistant output as GitHub-flavored markdown (tables, **bold**, lists,
// code, etc.) plus LaTeX math: remark-math parses `$…$` / `$$…$$`, rehype-katex
// renders it with KaTeX. Styling lives under the `.md` scope in styles.css so the
// element overrides stay out of the JSX. The plugins tolerate partial input, so
// streaming token-by-token re-parses fine.
export default function Markdown({ children }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
