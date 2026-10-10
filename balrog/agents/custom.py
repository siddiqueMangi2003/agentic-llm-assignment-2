import re

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """A planning agent that explicitly replans after repeating an action."""

    stall_threshold = 2

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None
        self.last_action = None
        self.repeat_count = 0

    def reset(self):
        super().reset()
        self.plan = None
        self.last_action = None
        self.repeat_count = 0

    def _record_action(self, action):
        if not action:
            return
        if action == self.last_action:
            self.repeat_count += 1
        else:
            self.last_action = action
            self.repeat_count = 1

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        if prev_action:
            self._record_action(prev_action)
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        stall_text = ""
        if self.repeat_count >= self.stall_threshold:
            stall_text = (
                f"\nThe previous action has been repeated {self.repeat_count} times. "
                "Treat this as possible stagnation and revise the plan before acting.\n"
            )

        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
ACTION: <your next action>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + stall_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan != "No changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        response = response._replace(reasoning=plan, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\nACTION:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"ACTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        action = action_match.group(1).strip() if action_match else None

        return plan, action
