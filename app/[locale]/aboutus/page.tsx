export default function AboutUsPage() {
  return (
    <div className="px-4 py-8 pb-16 md:px-0">
      <div className="dark:bg-secondary container mx-auto max-w-2xl space-y-6 rounded-md border border-gray-300 bg-white px-4 py-8 shadow-lg sm:px-8 dark:border-gray-700">
        <h1 className="mb-8 text-center text-4xl font-bold text-gray-900 dark:text-white">
          About us
        </h1>

        <div className="space-y-6">
          <div>
            <h3 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-gray-300">
              What is HackerGPT?
            </h3>
            <p className="text-lg text-gray-800 dark:text-gray-300">
              HackerGPT is an AI-powered assistant that is specifically designed
              for bug bounty. It is capable of providing comprehensive
              assistance and answering your hacking related questions using a
              vast dataset that includes detailed guides, extensive hacking
              write-ups, and in-depth bug bounty reports. In addition to that,
              HackerGPT enables direct interaction with hacking tools, which can
              significantly amplify your hacking endeavors.
            </p>
          </div>

          <div>
            <h3 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-gray-300">
              How does HackerGPT work?
            </h3>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              When you ask a question, it&apos;s sent to our server. We verify
              user authenticity and manage your question quota based on whether
              you&apos;re a free or pro user. We then search our hacking
              database for information that closely matches your question. If we
              find a strong match, we integrate it into the AI&apos;s response
              process. We then securely send your question to OpenRouter for
              processing without sending any personal information. Responses
              vary depending on the module:
            </p>
            <ul className="ml-8 list-disc space-y-2">
              <li>
                HackerGPT: A Mixtral 8x22B with a semantic search on our hacking
                data paired with our unique prompt.
              </li>
              <li>HackerGPT Pro: A Llama 3 70B with our unique prompt.</li>
              <li>
                GPT-4o: The latest and greatest from OpenAI, paired with our
                unique prompt.
              </li>
            </ul>
          </div>
          <div>
            <h3 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-gray-300">
              What Makes HackerGPT Special?
            </h3>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              HackerGPT is not just an AI that answers your hacking questions,
              it can also assist you in hacking using widely used open-source
              hacking tools. If you want to see all available tools, you can
              open the Plugin Store. Additionally, if you need a quick guide on
              using a specific tool such as Subfinder, select the tool and type{" "}
              <code>/subfinder -h</code>.
            </p>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              Below are some of the notable tools available with HackerGPT:
            </p>
            <ul className="ml-8 list-disc space-y-2">
              <li>
                <strong>
                  <a
                    href="https://github.com/projectdiscovery/subfinder"
                    className="text-blue-600 hover:text-blue-500 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    Subfinder
                  </a>
                </strong>{" "}
                is a subdomain discovery tool designed to enumerate and uncover
                valid subdomains of websites efficiently through passive online
                sources.
              </li>
              <li>
                <strong>
                  <a
                    href="https://github.com/projectdiscovery/katana"
                    className="text-blue-600 hover:text-blue-500 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    Katana
                  </a>
                </strong>{" "}
                is a next-generation crawling and spidering framework designed
                for robust, efficient web enumeration.
              </li>
              <li>
                <strong>
                  <a
                    href="https://github.com/projectdiscovery/naabu"
                    className="text-blue-600 hover:text-blue-500 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
                  >
                    Port Scanner
                  </a>
                </strong>{" "}
                is a high-speed port scanning tool, focused on delivering
                efficient and reliable network exploration.
              </li>
            </ul>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              Oh, and yes, you can effortlessly use these tools without typing
              complex commands — simply select the tool you want and describe in
              your own words what you need to do.
            </p>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              Along with these, there are more tools available with HackerGPT.
            </p>
          </div>
          <div>
            <h3 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-gray-300">
              Is HackerGPT Open Source?
            </h3>
            <p className="mb-4 text-lg text-gray-800 dark:text-gray-300">
              Absolutely! HackerGPT is committed to transparency and community
              collaboration. Our code is open source, allowing anyone to view,
              study, and understand how our software works. This also enables
              developers around the world to contribute to its development and
              improvement. Check out our GitHub repository for more details:{" "}
              <a
                href="https://github.com/Hacker-GPT/HackerGPT-2.0/"
                className="text-blue-600 hover:text-blue-500 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
              >
                HackerGPT on GitHub
              </a>
              .
            </p>
          </div>
          <div>
            <h3 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-gray-300">
              Need help or have questions?
            </h3>
            <p className="text-lg text-gray-800 dark:text-gray-300">
              We&apos;re here for you. Get in touch for any help, questions, or
              feedback at{" "}
              <a
                className="text-blue-600 hover:text-blue-500 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
                href="mailto:contact@hackerai.co"
              >
                contact@hackerai.co
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
