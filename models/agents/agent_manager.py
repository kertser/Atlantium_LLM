from typing import Dict, Any
import logging
from pathlib import Path
import json
import yaml
import copy
from models.agents.calculator import REDLibrary


class AgentManager:
    """Manages and orchestrates different agents in the system"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.agents: Dict[str, Any] = {}
        self.templates: Dict[str, Dict] = {}
        self._initialize_agents()
        self._load_templates()
        logging.info("AgentManager initialized with templates and agents")

    def _initialize_agents(self):
        """Initialize all available agents"""
        try:
            # Initialize calculator agent
            self.agents['calculator'] = REDLibrary(api_key=self.api_key)
            logging.info("Calculator agent initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing agents: {e}")
            raise

    def _load_templates(self):
        """Load all agent-specific templates"""
        templates_dir = Path(__file__).parent.parent / "templates" / "agents"

        for agent_dir in templates_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                for template_file in agent_dir.glob("*.yaml"):
                    try:
                        template_key = f"{agent_name}/{template_file.stem}"
                        with open(template_file, 'r', encoding='utf-8') as f:
                            self.templates[template_key] = yaml.safe_load(f)
                            logging.info(f"Loaded template: {template_key}")
                    except Exception as e:
                        logging.error(f"Error loading template {template_file}: {e}")

    async def _detect_calculator_content(self, text: str) -> Dict:
        """Detect if text contains calculator-related content"""
        try:
            if 'calculator' not in self.agents:
                logging.error("Calculator agent not initialized")
                return {"has_calculator_content": False}

            template = self.templates.get('calculator/detection')
            if not template:
                logging.error("Calculator detection template not found")
                return {"has_calculator_content": False}

            # Let the calculator agent handle the detection
            detection_result = self.agents['calculator'].detect_calculation_content(text, template)
            logging.info(f"Detection result: {detection_result}")
            return detection_result

        except Exception as e:
            logging.error(f"Error in calculator content detection: {e}")
            return {"has_calculator_content": False}

    async def detect_agent_requirements(self, text: str) -> Dict[str, Dict]:
        """Detect which agents are needed for the given text"""
        requirements = {}

        # Check calculator requirements
        if 'calculator' in self.agents:
            try:
                calc_detection = await self._detect_calculator_content(text)
                if isinstance(calc_detection, str):
                    try:
                        calc_detection = json.loads(calc_detection)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing calculator detection response: {e}")
                        return requirements

                if calc_detection.get('has_calculator_content'):
                    requirements['calculator'] = {
                        'parameters': calc_detection.get('parameters', {}),
                        'is_valid_system': calc_detection.get('is_valid_system', False),
                        'error_message': calc_detection.get('error_message'),
                        'extracted_text': calc_detection.get('extracted_text', '')
                    }
                    logging.info(f"Detected calculator requirements: {requirements['calculator']}")
            except Exception as e:
                logging.error(f"Error in calculator detection: {e}")

        return requirements

    async def process_with_agents(self, text: str, requirements: Dict[str, Dict]) -> Dict[str, Any]:
        """Process text with required agents"""
        results = {}

        if 'calculator' in requirements:
            try:
                calc_requirements = requirements['calculator']

                # If system is invalid, return the error message
                if not calc_requirements.get('is_valid_system', False):
                    return {
                        'calculator': {
                            'error': calc_requirements.get('error_message', 'Invalid system type'),
                            'parameters': calc_requirements.get('parameters', {}),
                            'extracted_text': calc_requirements.get('extracted_text', '')
                        }
                    }

                # Get parameters from the correct location
                calc_params = calc_requirements.get('parameters', {})
                if not calc_params:
                    raise ValueError("No calculator parameters found")

                calculator_params = {
                    "system_type": calc_params['system_type'],
                    "flow": calc_params['flow'],
                    "uvt": calc_params['uvt'],
                    "power_settings": {},
                    "efficiency_settings": {
                        "all_lamps": 80.0
                    }
                }

                # Handle power settings
                if 'power_settings' in calc_params:
                    power_settings = calc_params['power_settings']

                    # Handle specific lamp powers first
                    specific_powers = {
                        str(key.split('_')[1]): float(value)
                        for key, value in power_settings.items()
                        if key.startswith('lamp_')
                    }

                    # Set the base power setting
                    if 'all_other_lamps' in power_settings:
                        calculator_params["power_settings"]["all_lamps"] = float(power_settings['all_other_lamps'])
                    elif 'all_lamps' in power_settings:
                        calculator_params["power_settings"]["all_lamps"] = float(power_settings['all_lamps'])
                    else:
                        calculator_params["power_settings"]["all_lamps"] = 100.0

                    # Add specific lamp powers if present
                    if specific_powers:
                        calculator_params["power_settings"]["specific_lamps"] = specific_powers

                # Handle efficiency settings
                if 'lamp_efficiencies' in calc_params:
                    specific_efficiencies = {
                        str(key.split('_')[1]): float(value)
                        for key, value in calc_params['lamp_efficiencies'].items()
                        if key.startswith('lamp_')
                    }
                    if specific_efficiencies:
                        calculator_params["efficiency_settings"]["specific_lamps"] = specific_efficiencies

                # Ensure the specific_lamps setting is preserved in the nested result
                orig_calculator_params = copy.deepcopy(calculator_params)

                logging.debug(f"Sending calculator parameters: {json.dumps(calculator_params, indent=2)}")
                calc_result = self.agents['calculator'].process_query(json.dumps(calculator_params))

                if calc_result and 'error' not in calc_result:
                    # Preserve the original parameters in the result
                    if isinstance(calc_result.get('result'), dict):
                        calc_result['result']['original_parameters'] = orig_calculator_params

                    results['calculator'] = calc_result
                    logging.info(f"Calculator processing successful: {calc_result.get('result')} mJ/cm²")
                else:
                    results['calculator'] = {
                        'error': calc_result.get('error', 'Unknown calculator error'),
                        'parameters': calc_requirements.get('parameters', {}),
                        'extracted_text': calc_requirements.get('extracted_text', '')
                    }
                    logging.error(f"Calculator error: {calc_result.get('error')}")

            except Exception as e:
                logging.error(f"Error in calculator processing: {str(e)}")
                logging.error("Stack trace:", exc_info=True)
                results['calculator'] = {
                    'error': str(e),
                    'parameters': calc_requirements.get('parameters', {}) if 'calc_requirements' in locals() else {},
                    'extracted_text': calc_requirements.get('extracted_text',
                                                            '') if 'calc_requirements' in locals() else ''
                }

        return results

    async def aggregate_responses(self, rag_response: str, agent_results: Dict[str, Any]) -> str:
        """Combine RAG response with agent results using templates"""
        if not agent_results:
            logging.debug("No agent results to aggregate")
            return rag_response

        try:
            if 'calculator' in agent_results:
                calc_result = agent_results['calculator']

                # Check if there's an error
                if 'error' in calc_result:
                    # Format error message
                    formatted_response = [
                        "# ❌ Calculation Error",
                        "",
                        f"**Error**: {calc_result['error']}",
                        "",
                        "** Provided Parameters: **"
                    ]

                    # Add provided parameters if available
                    params = calc_result.get('parameters', {})
                    if params:
                        formatted_response.extend([
                            f"• **System**: {params.get('system_type', 'N/A')}",
                            f"• **Flow Rate**: {params.get('flow', 'N/A')} m³/h",
                            f"• **UVT**: {params.get('uvt', 'N/A')}%"
                        ])

                        # Add power settings if present
                        if 'power_settings' in params:
                            power_settings = params['power_settings']
                            power_str = f"• **Power**: {power_settings.get('all_lamps', 'N/A')}%"
                            formatted_response.append(power_str)

                    # Add valid systems list if it's a system type error
                    if "Unsupported system type" in calc_result['error']:
                        formatted_response.extend([
                            "",
                            "**Valid System Types:**",
                            f"<small>{', '.join(self.agents['calculator'].supported_systems)}</small>"
                        ])

                    return "\n".join(formatted_response)

                # Handle successful calculation
                # Extract the calculation results and parameters
                result_data = calc_result.get('result', {}).get('result', {})
                details = calc_result.get('result', {}).get('details', {})
                parameters = details.get('parameters', {})
                lamp_settings = details.get('lamp_settings', {})

                # Format response with underlines
                formatted_response = [
                    "# Calculation Results",
                    "",
                    "System Parameters:",
                    f"• **System**: {details.get('system_type', 'Unknown')}",
                    f"• **Number of Lamps**: {details.get('number_of_lamps', 'N/A')}",
                    f"• **Lamp Power**: {details.get('lamp_power_watts', 'N/A')} Watts",
                    f"• **Flow Rate**: {parameters.get('flow', 'N/A')} m³/h",
                    f"• **UVT**: {parameters.get('uvt', 'N/A')}%-1cm",
                    "",
                    "Lamp Settings:",
                    "**Power Settings**:"
                ]

                # Format power settings
                power_settings = lamp_settings.get('power', [])
                if power_settings:
                    unique_powers = set(power_settings)
                    if len(unique_powers) == 1:
                        # All lamps at same power
                        formatted_response.append(f"• All lamps at {power_settings[0]}% power")
                    else:
                        # Format individual lamp settings
                        for i, power in enumerate(power_settings, 1):
                            formatted_response.append(f"• Lamp {i}: {power}% power")

                formatted_response.extend([
                    "",
                    "**Efficiency Settings**:"
                ])

                # Format efficiency settings
                efficiency_settings = lamp_settings.get('efficiency', [])
                if efficiency_settings:
                    unique_efficiencies = set(efficiency_settings)
                    if len(unique_efficiencies) == 1:
                        # All lamps at same efficiency
                        formatted_response.append(f"• All lamps at {efficiency_settings[0]}% efficiency")
                    else:
                        # Format individual lamp efficiencies
                        for i, eff in enumerate(efficiency_settings, 1):
                            formatted_response.append(f"• Lamp {i}: {eff}% efficiency")

                # Add results section
                formatted_response.extend([
                    "",
                    "Results:",
                    f"• **RED Value**: {result_data} mJ/cm²",
                    "",
                    "✅ Calculation completed successfully"
                ])

                return "\n".join(formatted_response)

            return rag_response

        except Exception as e:
            logging.error(f"Error in response aggregation: {e}")
            return rag_response

    def _format_settings(self, settings: list) -> str:
        """Format power or efficiency settings"""
        if not settings:
            return ""
        if len(set(settings)) == 1:
            return f"All lamps: {settings[0]}%"
        return "\n".join(f"- Lamp {i + 1}: {value}%" for i, value in enumerate(settings))

    def _default_aggregation(self, rag_response: str, agent_results: Dict[str, Any]) -> str:
        """Fallback aggregation method when template is unavailable"""
        if 'calculator' in agent_results:
            calc_result = agent_results['calculator']
            calc_section = [
                "## Calculation Results",
                f"- System: {calc_result.get('details', {}).get('system_type', 'Unknown')}",
                f"- Flow Rate: {calc_result.get('details', {}).get('parameters', {}).get('flow', 'N/A')} m³/h",
                f"- UVT: {calc_result.get('details', {}).get('parameters', {}).get('uvt', 'N/A')}%-1cm",
                f"- RED Value: {calc_result.get('result', 'N/A')} mJ/cm²",
                "✅ Calculation completed successfully"
            ]

            if rag_response.strip():
                return f"{rag_response}\n\n{'\n'.join(calc_section)}"
            return "\n".join(calc_section)

        return rag_response
