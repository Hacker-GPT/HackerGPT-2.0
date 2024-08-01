type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

type LLMProvider = {
  provider: string;
  model: string;
};

const DEFAULT_MODEL = 'gpt-4o';

async function selectModel(messages: Message[], llmProviders: LLMProvider[]): Promise<string> {
  if (!process.env.NOTDIAMOND_API_KEY) {
    return DEFAULT_MODEL;
  }

  try {
    const notDiamondResponse = await fetch('https://not-diamond-server.onrender.com/v2/modelRouter/modelSelect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.NOTDIAMOND_API_KEY}`
      },
      body: JSON.stringify({ messages, llm_providers: llmProviders })
    });

    if (!notDiamondResponse.ok) {
      const errorText = await notDiamondResponse.text();
      console.error(`NotDiamond API error (${notDiamondResponse.status}): ${errorText}`);
      return DEFAULT_MODEL;
    }

    const modelSelection = await notDiamondResponse.json();
    return modelSelection.providers[0].model;
  } catch (error) {
    console.error('NotDiamond error:', error);
    return DEFAULT_MODEL;
  }
}

function transformMessages(messages: any[]): Message[] {
  return messages.map(msg => ({
    role: msg.role as "user" | "assistant" | "system",
    content: typeof msg.content === 'string' ? msg.content : msg.content.map((c: { type: string; text: any; }) => c.type === 'text' ? c.text : '').join(' ')
  }));
}

export async function getSelectedModel(messages: any[]): Promise<string> {
  const transformedMessages = transformMessages(messages);
  const llmProviders = [
    { provider: 'openai', model: 'gpt-4o' },
    { provider: 'openai', model: 'gpt-4o-mini' }
  ];

  return selectModel(transformedMessages, llmProviders);
}