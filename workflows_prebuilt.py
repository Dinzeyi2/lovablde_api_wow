"""
Pre-built Workflow Templates
7 complete workflows for different app types
"""

PREBUILT_WORKFLOWS = {
    # ==================== 1. E-COMMERCE COMPLETE CHECKOUT ====================
    
    "ecommerce_checkout": {
        "name": "E-commerce Complete Checkout",
        "description": "Complete purchase workflow with fraud detection, pricing, tax, and shipping",
        "category": "ecommerce",
        "steps": [
            {
                "step_id": "validate_inventory",
                "type": "algorithm",
                "algorithm": "inventory-optimization",
                "input": {
                    "current_stock": "{{product_stock}}",
                    "daily_demand": "{{avg_daily_sales}}",
                    "holding_cost_percent": 15,
                    "ordering_cost": 50,
                    "lead_time_days": 5,
                    "service_level": 0.95,
                    "demand_variability": 2
                },
                "conditions": [
                    {
                        "if": "output.current_stock < 1",
                        "then": "stop_workflow",
                        "return": {
                            "status": "out_of_stock",
                            "message": "Product unavailable"
                        }
                    }
                ]
            },
            {
                "step_id": "check_fraud",
                "type": "algorithm",
                "algorithm": "fraud-detection",
                "input": {
                    "transaction_amount": "{{total_amount}}",
                    "user_location": "{{country_code}}",
                    "device_fingerprint": "{{device_id}}",
                    "account_age_days": "{{account_age}}",
                    "transaction_time": "{{current_hour}}",
                    "phone_verified": "{{phone_verified}}",
                    "email_domain": "{{email_domain}}"
                },
                "conditions": [
                    {
                        "if": "output.risk_score > 0.7",
                        "then": "stop_workflow",
                        "return": {
                            "status": "fraud_blocked",
                            "fraud_score": "{{steps.check_fraud.output.risk_score}}",
                            "reason": "High fraud risk detected"
                        }
                    }
                ]
            },
            {
                "step_id": "calculate_dynamic_price",
                "type": "algorithm",
                "algorithm": "dynamic-pricing",
                "input": {
                    "base_price": "{{base_price}}",
                    "competitor_prices": "{{competitor_prices}}",
                    "inventory_level": "{{inventory_percentage}}",
                    "demand_score": 0.5,
                    "time_of_day": "{{current_hour}}",
                    "day_of_week": "{{current_day}}",
                    "cost": "{{product_cost}}"
                }
            },
            {
                "step_id": "calculate_tax",
                "type": "algorithm",
                "algorithm": "tax-calculator-us",
                "input": {
                    "income": "{{steps.calculate_dynamic_price.output.recommended_price}}",
                    "filing_status": "single",
                    "deductions": 0
                }
            },
            {
                "step_id": "calculate_shipping",
                "type": "algorithm",
                "algorithm": "shipping-calculator",
                "input": {
                    "weight_lbs": "{{product_weight}}",
                    "length_inches": "{{package_length}}",
                    "width_inches": "{{package_width}}",
                    "height_inches": "{{package_height}}",
                    "origin_zip": "{{warehouse_zip}}",
                    "destination_zip": "{{shipping_zip}}",
                    "carrier": "usps"
                }
            },
            {
                "step_id": "calculate_total",
                "type": "custom_logic",
                "operation": "sum",
                "fields": [
                    "{{steps.calculate_dynamic_price.output.recommended_price}}",
                    "{{steps.calculate_tax.output.federal_tax}}",
                    "{{steps.calculate_shipping.output.total_cost}}"
                ]
            }
        ],
        "output_format": {
            "order_total": "{{steps.calculate_total.output}}",
            "breakdown": {
                "subtotal": "{{steps.calculate_dynamic_price.output.recommended_price}}",
                "tax": "{{steps.calculate_tax.output.federal_tax}}",
                "shipping": "{{steps.calculate_shipping.output.total_cost}}",
                "fraud_score": "{{steps.check_fraud.output.risk_score}}"
            },
            "inventory_status": "{{steps.validate_inventory.output.needs_reorder}}"
        }
    },
    
    # ==================== 2. FINTECH LOAN APPROVAL ====================
    
    "fintech_loan_approval": {
        "name": "Fintech Loan Approval Workflow",
        "description": "Complete loan application processing with credit scoring, fraud detection, and risk analysis",
        "category": "fintech",
        "steps": [
            {
                "step_id": "verify_identity",
                "type": "algorithm",
                "algorithm": "fraud-detection",
                "input": {
                    "transaction_amount": "{{loan_amount}}",
                    "user_location": "{{country_code}}",
                    "device_fingerprint": "{{device_id}}",
                    "account_age_days": "{{account_age}}",
                    "transaction_time": "{{current_hour}}"
                },
                "conditions": [
                    {
                        "if": "output.risk_score > 0.6",
                        "then": "stop_workflow",
                        "return": {
                            "status": "rejected",
                            "reason": "Identity verification failed"
                        }
                    }
                ]
            },
            {
                "step_id": "calculate_credit_score",
                "type": "algorithm",
                "algorithm": "credit-scoring",
                "input": {
                    "payment_history_score": "{{payment_history}}",
                    "credit_utilization": "{{credit_utilization}}",
                    "credit_age_years": "{{credit_age}}",
                    "num_accounts": "{{num_accounts}}",
                    "hard_inquiries": "{{hard_inquiries}}",
                    "derogatory_marks": "{{derogatory_marks}}",
                    "has_credit_card": "{{has_credit_card}}",
                    "has_auto_loan": "{{has_auto_loan}}",
                    "has_mortgage": "{{has_mortgage}}",
                    "has_student_loan": "{{has_student_loan}}"
                },
                "conditions": [
                    {
                        "if": "output.credit_score < 600",
                        "then": "stop_workflow",
                        "return": {
                            "status": "rejected",
                            "reason": "Credit score below minimum requirement",
                            "credit_score": "{{steps.calculate_credit_score.output.credit_score}}"
                        }
                    }
                ]
            },
            {
                "step_id": "calculate_monthly_payment",
                "type": "algorithm",
                "algorithm": "loan-amortization",
                "input": {
                    "loan_amount": "{{loan_amount}}",
                    "annual_rate": "{{interest_rate}}",
                    "loan_term_years": "{{loan_term}}",
                    "payment_frequency": "monthly"
                }
            },
            {
                "step_id": "risk_analysis",
                "type": "algorithm",
                "algorithm": "monte-carlo-simulation",
                "input": {
                    "initial_value": "{{loan_amount}}",
                    "expected_return": 0.08,
                    "volatility": 0.15,
                    "time_horizon_years": "{{loan_term}}",
                    "simulations": 1000
                }
            },
            {
                "step_id": "final_decision",
                "type": "custom_logic",
                "operation": "max",
                "fields": [
                    "{{steps.calculate_credit_score.output.credit_score}}"
                ]
            }
        ],
        "output_format": {
            "status": "approved",
            "loan_amount": "{{loan_amount}}",
            "monthly_payment": "{{steps.calculate_monthly_payment.output.monthly_payment}}",
            "total_interest": "{{steps.calculate_monthly_payment.output.total_interest}}",
            "credit_score": "{{steps.calculate_credit_score.output.credit_score}}",
            "approval_tier": "{{steps.calculate_credit_score.output.interest_tier}}",
            "risk_analysis": {
                "probability_of_loss": "{{steps.risk_analysis.output.probability_of_loss}}",
                "expected_value": "{{steps.risk_analysis.output.mean_outcome}}"
            }
        }
    },
    
    # ==================== 3. MARKETING LEAD QUALIFICATION ====================
    
    "marketing_lead_qualification": {
        "name": "Marketing Lead Qualification Workflow",
        "description": "Score leads, predict churn risk, calculate CLV, and segment customers",
        "category": "marketing",
        "steps": [
            {
                "step_id": "score_lead",
                "type": "algorithm",
                "algorithm": "lead-scoring",
                "input": {
                    "email_opens": "{{email_opens}}",
                    "email_clicks": "{{email_clicks}}",
                    "page_views": "{{page_views}}",
                    "time_on_site": "{{time_on_site}}",
                    "downloads": "{{downloads}}",
                    "webinar_attended": "{{webinar_attended}}",
                    "company_size": "{{company_size}}",
                    "job_title": "{{job_title}}",
                    "industry": "{{industry}}",
                    "budget_indicated": "{{budget_indicated}}",
                    "has_budget": "{{has_budget}}",
                    "has_authority": "{{has_authority}}",
                    "has_need": "{{has_need}}",
                    "has_timeline": "{{has_timeline}}"
                },
                "conditions": [
                    {
                        "if": "output.lead_score < 40",
                        "then": "stop_workflow",
                        "return": {
                            "status": "cold_lead",
                            "action": "add_to_nurture_campaign",
                            "lead_score": "{{steps.score_lead.output.lead_score}}"
                        }
                    }
                ]
            },
            {
                "step_id": "predict_churn",
                "type": "algorithm",
                "algorithm": "churn-prediction",
                "input": {
                    "days_since_last_activity": "{{days_since_last_activity}}",
                    "total_purchases": "{{total_purchases}}",
                    "avg_purchase_value": "{{avg_purchase_value}}",
                    "support_tickets": "{{support_tickets}}",
                    "account_age_months": "{{account_age_months}}",
                    "login_frequency": "{{login_frequency}}",
                    "feature_usage_score": "{{feature_usage_score}}",
                    "payment_failures": "{{payment_failures}}",
                    "nps_score": "{{nps_score}}"
                }
            },
            {
                "step_id": "calculate_clv",
                "type": "algorithm",
                "algorithm": "customer-lifetime-value",
                "input": {
                    "avg_purchase_value": "{{avg_purchase_value}}",
                    "purchase_frequency_per_year": "{{purchase_frequency}}",
                    "customer_lifespan_years": 3,
                    "gross_margin_percent": 30,
                    "discount_rate": 0.10
                }
            },
            {
                "step_id": "segment_customer",
                "type": "algorithm",
                "algorithm": "customer-segmentation",
                "input": {
                    "days_since_last_purchase": "{{days_since_last_activity}}",
                    "total_purchases": "{{total_purchases}}",
                    "total_spent": "{{total_spent}}"
                }
            }
        ],
        "output_format": {
            "lead_quality": "{{steps.score_lead.output.quality}}",
            "lead_score": "{{steps.score_lead.output.lead_score}}",
            "priority": "{{steps.score_lead.output.priority}}",
            "churn_risk": {
                "probability": "{{steps.predict_churn.output.churn_probability}}",
                "risk_level": "{{steps.predict_churn.output.risk_level}}",
                "recommended_action": "{{steps.predict_churn.output.recommended_action}}"
            },
            "customer_lifetime_value": "{{steps.calculate_clv.output.lifetime_value_net}}",
            "recommended_max_cac": "{{steps.calculate_clv.output.recommended_max_cac}}",
            "customer_segment": "{{steps.segment_customer.output.segment}}",
            "next_action": "{{steps.segment_customer.output.recommended_action}}"
        }
    },
    
    # ==================== 4. LOGISTICS DELIVERY OPTIMIZATION ====================
    
    "logistics_delivery_optimization": {
        "name": "Logistics Delivery Optimization Workflow",
        "description": "Optimize delivery routes, calculate shipping costs, and forecast demand",
        "category": "logistics",
        "steps": [
            {
                "step_id": "forecast_demand",
                "type": "algorithm",
                "algorithm": "demand-forecasting",
                "input": {
                    "historical_sales": "{{historical_sales}}",
                    "forecast_periods": 7,
                    "seasonality_period": 7,
                    "alpha": 0.3,
                    "beta": 0.1,
                    "gamma": 0.3
                }
            },
            {
                "step_id": "optimize_routes",
                "type": "algorithm",
                "algorithm": "route-optimization",
                "input": {
                    "locations": "{{delivery_locations}}",
                    "start_location": "{{warehouse_location}}",
                    "return_to_start": True,
                    "avg_speed_kmh": 50,
                    "stop_time_minutes": 15
                }
            },
            {
                "step_id": "calculate_shipping_costs",
                "type": "algorithm",
                "algorithm": "shipping-calculator",
                "input": {
                    "weight_lbs": "{{total_weight}}",
                    "length_inches": "{{package_length}}",
                    "width_inches": "{{package_width}}",
                    "height_inches": "{{package_height}}",
                    "origin_zip": "{{warehouse_zip}}",
                    "destination_zip": "{{customer_zip}}",
                    "carrier": "fedex"
                }
            },
            {
                "step_id": "inventory_check",
                "type": "algorithm",
                "algorithm": "inventory-optimization",
                "input": {
                    "current_stock": "{{current_stock}}",
                    "daily_demand": "{{avg_daily_sales}}",
                    "holding_cost_percent": 15,
                    "ordering_cost": 50,
                    "lead_time_days": 7,
                    "service_level": 0.95,
                    "demand_variability": 3
                }
            }
        ],
        "output_format": {
            "demand_forecast": "{{steps.forecast_demand.output.forecast}}",
            "forecast_trend": "{{steps.forecast_demand.output.trend}}",
            "optimized_route": {
                "total_distance_km": "{{steps.optimize_routes.output.total_distance_km}}",
                "total_time_hours": "{{steps.optimize_routes.output.total_time_hours}}",
                "num_stops": "{{steps.optimize_routes.output.num_stops}}",
                "route": "{{steps.optimize_routes.output.optimized_route}}"
            },
            "shipping_cost": "{{steps.calculate_shipping_costs.output.total_cost}}",
            "estimated_delivery_days": "{{steps.calculate_shipping_costs.output.estimated_delivery_days}}",
            "inventory_status": {
                "needs_reorder": "{{steps.inventory_check.output.needs_reorder}}",
                "reorder_quantity": "{{steps.inventory_check.output.economic_order_quantity}}",
                "urgency": "{{steps.inventory_check.output.urgency}}"
            }
        }
    },
    
    # ==================== 5. HEALTHCARE PATIENT TRIAGE ====================
    
    "healthcare_patient_assessment": {
        "name": "Healthcare Patient Assessment Workflow",
        "description": "BMI calculation, medication interaction check, and health risk assessment",
        "category": "healthcare",
        "steps": [
            {
                "step_id": "calculate_bmi",
                "type": "algorithm",
                "algorithm": "bmi-calculator",
                "input": {
                    "weight_kg": "{{weight_kg}}",
                    "height_m": "{{height_m}}",
                    "age": "{{age}}",
                    "sex": "{{gender}}"
                }
            },
            {
                "step_id": "check_medications",
                "type": "algorithm",
                "algorithm": "medication-interaction-checker",
                "input": {
                    "medications": "{{current_medications}}"
                },
                "conditions": [
                    {
                        "if": "output.interaction_count > 0",
                        "then": "continue",
                        "return": {
                            "warning": "Drug interactions detected - consult pharmacist"
                        }
                    }
                ]
            }
        ],
        "output_format": {
            "bmi": "{{steps.calculate_bmi.output.bmi}}",
            "bmi_category": "{{steps.calculate_bmi.output.category}}",
            "health_risk": "{{steps.calculate_bmi.output.health_risk}}",
            "recommendations": "{{steps.calculate_bmi.output.recommendations}}",
            "medication_warnings": {
                "has_interactions": "{{steps.check_medications.output.has_interactions}}",
                "interactions": "{{steps.check_medications.output.interactions}}",
                "recommendation": "{{steps.check_medications.output.recommendation}}"
            }
        }
    },
    
    # ==================== 6. CONTENT MODERATION ====================
    
    "content_moderation_pipeline": {
        "name": "Content Moderation Pipeline",
        "description": "Multi-layer content safety checking with plagiarism detection and sentiment analysis",
        "category": "content",
        "steps": [
            {
                "step_id": "moderate_content",
                "type": "algorithm",
                "algorithm": "content-moderation",
                "input": {
                    "text": "{{content_text}}"
                },
                "conditions": [
                    {
                        "if": "output.safety_score < 50",
                        "then": "stop_workflow",
                        "return": {
                            "status": "rejected",
                            "reason": "Content safety violation",
                            "flags": "{{steps.moderate_content.output.flags}}"
                        }
                    }
                ]
            },
            {
                "step_id": "check_plagiarism",
                "type": "algorithm",
                "algorithm": "plagiarism-detector",
                "input": {
                    "text": "{{content_text}}",
                    "reference_texts": "{{reference_texts}}"
                },
                "conditions": [
                    {
                        "if": "output.is_plagiarized == True",
                        "then": "stop_workflow",
                        "return": {
                            "status": "rejected",
                            "reason": "Plagiarism detected",
                            "similarity_score": "{{steps.check_plagiarism.output.similarity_score}}"
                        }
                    }
                ]
            },
            {
                "step_id": "analyze_sentiment",
                "type": "algorithm",
                "algorithm": "sentiment-analysis",
                "input": {
                    "text": "{{content_text}}"
                }
            },
            {
                "step_id": "check_readability",
                "type": "algorithm",
                "algorithm": "readability-score",
                "input": {
                    "text": "{{content_text}}"
                }
            }
        ],
        "output_format": {
            "status": "approved",
            "safety_score": "{{steps.moderate_content.output.safety_score}}",
            "action": "{{steps.moderate_content.output.action}}",
            "plagiarism_check": {
                "is_plagiarized": "{{steps.check_plagiarism.output.is_plagiarized}}",
                "similarity": "{{steps.check_plagiarism.output.similarity_score}}"
            },
            "sentiment": "{{steps.analyze_sentiment.output.sentiment}}",
            "readability": {
                "flesch_score": "{{steps.check_readability.output.flesch_reading_ease}}",
                "difficulty": "{{steps.check_readability.output.difficulty}}",
                "target_audience": "{{steps.check_readability.output.target_audience}}"
            }
        }
    },
    
    # ==================== 7. SECURITY THREAT DETECTION ====================
    
    "security_threat_detection": {
        "name": "Security Threat Detection Workflow",
        "description": "Multi-layer security scanning for SQL injection, XSS, and password strength",
        "category": "security",
        "steps": [
            {
                "step_id": "check_sql_injection",
                "type": "algorithm",
                "algorithm": "sql-injection-detector",
                "input": {
                    "user_input": "{{user_input}}"
                },
                "conditions": [
                    {
                        "if": "output.risk_score > 70",
                        "then": "stop_workflow",
                        "return": {
                            "status": "blocked",
                            "threat": "SQL Injection",
                            "risk_score": "{{steps.check_sql_injection.output.risk_score}}"
                        }
                    }
                ]
            },
            {
                "step_id": "check_xss",
                "type": "algorithm",
                "algorithm": "xss-attack-detector",
                "input": {
                    "user_input": "{{user_input}}"
                },
                "conditions": [
                    {
                        "if": "output.risk_score > 60",
                        "then": "stop_workflow",
                        "return": {
                            "status": "blocked",
                            "threat": "XSS Attack",
                            "risk_score": "{{steps.check_xss.output.risk_score}}"
                        }
                    }
                ]
            },
            {
                "step_id": "validate_password",
                "type": "algorithm",
                "algorithm": "password-strength",
                "input": {
                    "password": "{{password}}"
                },
                "conditions": [
                    {
                        "if": "output.score < 40",
                        "then": "stop_workflow",
                        "return": {
                            "status": "rejected",
                            "reason": "Password too weak",
                            "strength": "{{steps.validate_password.output.strength}}",
                            "feedback": "{{steps.validate_password.output.feedback}}"
                        }
                    }
                ]
            }
        ],
        "output_format": {
            "status": "safe",
            "sql_injection_risk": "{{steps.check_sql_injection.output.risk_score}}",
            "xss_risk": "{{steps.check_xss.output.risk_score}}",
            "password_strength": "{{steps.validate_password.output.strength}}",
            "password_score": "{{steps.validate_password.output.score}}",
            "sanitized_input": "{{steps.check_sql_injection.output.sanitized_input}}"
        }
    }
}

# Workflow categories
WORKFLOW_CATEGORIES = {
    'ecommerce': ['ecommerce_checkout'],
    'fintech': ['fintech_loan_approval'],
    'marketing': ['marketing_lead_qualification'],
    'logistics': ['logistics_delivery_optimization'],
    'healthcare': ['healthcare_patient_assessment'],
    'content': ['content_moderation_pipeline'],
    'security': ['security_threat_detection']
}
