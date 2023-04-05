import { Tool } from "langchain/agents";
import { ConversationChain } from "langchain/chains";

export class ConversationTool extends Tool {
	name = "respond";

	description = "Useful for responding to the user and expressing your thoughts. The input tot his tool should be whatever the user said.";

	constructor(private chain: ConversationChain) {
	  super();
	}
  
	async _call(input: string): Promise<string> {
	  const response = await this.chain.call({input});
	  return response.response;
	}
  }