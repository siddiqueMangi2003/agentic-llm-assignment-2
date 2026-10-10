import re

from balrog.agents.custom import CustomAgent


class StallReflectionAgent(CustomAgent):
    """Critique a candidate plan and action only after repeated actions indicate a stall."""

    def act(self, obs, prev_action=None):
        candidate = super().act(obs, prev_action=prev_action)
        if self.repeat_count < self.stall_threshold:
            return candidate

        messages = self.prompt_builder.get_prompt()
        reflection_instructions = f"""
You are now a critic reviewing an agent that may be stuck.
Candidate plan: {self.plan}
Candidate action: {candidate.completion}

Identify why the candidate may repeat the failure, then return a revised plan and exactly one next action.
Format your answer as:
REFLECTION: <brief diagnosis>
PLAN: <revised plan>
ACTION: <next action>
        """.strip()
        messages[-1].content += "\n\n" + reflection_instructions

        reviewed = self.client.generate(messages)
        plan, action = self._extract_plan_and_action(reviewed.completion)
        reflection_match = re.search(
            r"REFLECTION:\s*(.*?)(?=\nPLAN:|\Z)", reviewed.completion, re.IGNORECASE | re.DOTALL
        )
        reflection = reflection_match.group(1).strip() if reflection_match else "No reflection returned."
        if plan != "No changes to the plan.":
            self.plan = plan

        return reviewed._replace(
            completion=action,
            reasoning=f"REFLECTION: {reflection}\nPLAN: {plan}",
            input_tokens=candidate.input_tokens + reviewed.input_tokens,
            output_tokens=candidate.output_tokens + reviewed.output_tokens,
        )
