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
        calc_requirements = []

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

    def _format_settings(self, settings: list) -> str:
        """Format power or efficiency settings"""
        if not settings:
            return ""
        if len(set(settings)) == 1:
            return f"• All lamps: {settings[0]}%"
        return "\n".join(f"• Lamp {i + 1}: {value}%" for i, value in enumerate(settings))

    async def aggregate_responses(self, rag_response: str, agent_results: Dict[str, Any]) -> str:
        if not agent_results or 'calculator' not in agent_results:
            return rag_response

        try:
            calc_result = agent_results['calculator']

            # Start building response
            formatted_response = ["<strong><u>Calculation Results:</u></strong>"]

            if 'error' in calc_result:
                error_data = calc_result['error']
                params = calc_result.get('parameters', {})

                if error_data.get('type') == 'validation':
                    validation_errors = error_data.get('errors', {})
                    formatted_response.extend([
                        "",
                        "** Parameter Validation Error **",
                        ""
                    ])

                    for param, error_info in validation_errors.items():
                        formatted_response.extend([
                            f"• ❌ {param.upper()}: {error_info['value']} {error_info['unit']} is out of range",
                            f"• ℹ️ Valid range: {error_info['min']} - {error_info['max']} {error_info['unit']}",
                            ""
                        ])

                    formatted_response.extend([
                        "** System Parameters **",
                        f"• System: {params.get('system_type', 'Unknown')}",
                        f"• Flow Rate: {params.get('flow', 'N/A')} m³/h",
                        f"• UVT: {params.get('uvt', 'N/A')}%-1cm",
                        "",
                        "❌ ** Calculation failed due to parameter validation errors **"
                    ])
                else:
                    formatted_response.extend([
                        "",
                        f"❌ {error_data.get('message', 'Unknown error occurred')}"
                    ])
            else:
                result = calc_result.get('result', {})
                details = result.get('details', {})
                parameters = details.get('parameters', {})
                lamp_settings = details.get('lamp_settings', {})

                # System Parameters section
                formatted_response.extend([
                    "",
                    "** System Parameters **",
                    f"• System: {details.get('system_type', 'Unknown')}",
                    f"• Number of Lamps: {details.get('number_of_lamps', 'N/A')}",
                    f"• Lamp Power: {details.get('lamp_power_watts', 'N/A')} W",
                    f"• Flow Rate: {parameters.get('flow', 'N/A')} m³/h",
                    f"• UVT: {parameters.get('uvt', 'N/A')}%-1cm"
                ])

                # Lamp Settings section
                power_settings = lamp_settings.get('power', [])
                efficiency_settings = lamp_settings.get('efficiency', [])

                if power_settings or efficiency_settings:
                    formatted_response.extend([
                        "",
                        "<strong> Lamp Settings </strong><br>"
                    ])

                    if power_settings:
                        formatted_response.extend([
                            "<em>ℹ️Power Settings:</em><br>",
                            self._format_settings(power_settings)
                        ])

                    if efficiency_settings:
                        formatted_response.extend([
                            "",
                            "<em>ℹ️Efficiency Settings:</em><br>",
                            self._format_settings(efficiency_settings)
                        ])

                # Results section
                formatted_response.extend([
                    "",
                    "<u><strong>Results:</strong></u><br>",
                    f"ℹ️<bold>RED Value:</bold> <em><bold>{result.get('result', 'N/A')} </bold></em>[mJ/cm²]",
                    "",
                    "✅ Calculation completed successfully"
                ])

            return "\n".join(formatted_response)

        except Exception as e:
            logging.error(f"Error in response aggregation: {e}", exc_info=True)
            return "** Error **\n\n> ❌ An error occurred while processing the calculation"

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
                return f"{rag_response}\n\n" + "\n".join(calc_section)
            return "\n".join(calc_section)

        return rag_response
