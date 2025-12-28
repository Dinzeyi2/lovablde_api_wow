/**
 * Example: Fraud Detection in Lovable.dev E-commerce App
 * 
 * This shows how to integrate AlgoAPI fraud detection
 * into your Lovable.dev checkout flow
 */

import React, { useState } from 'react';
import { useAlgoAPI } from '@algoapi/client';

function CheckoutPage() {
  const { detectFraud } = useAlgoAPI(process.env.REACT_APP_ALGOAPI_KEY);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fraudResult, setFraudResult] = useState(null);

  const handlePayment = async (paymentData) => {
    setIsProcessing(true);

    // Check for fraud before processing payment
    const fraudCheck = await detectFraud({
      transaction_amount: paymentData.amount,
      user_location: paymentData.country,
      device_fingerprint: getDeviceFingerprint(),
      account_age_days: calculateAccountAge(paymentData.userId),
      transaction_time: new Date().getHours()
    });

    setFraudResult(fraudCheck.result);

    if (fraudCheck.result.recommendation === 'block') {
      alert('Transaction blocked due to suspicious activity');
      setIsProcessing(false);
      return;
    }

    if (fraudCheck.result.recommendation === 'review') {
      // Send to manual review queue
      await sendToReviewQueue(paymentData, fraudCheck.result);
      alert('Payment is under review');
      setIsProcessing(false);
      return;
    }

    // Process payment normally
    await processPayment(paymentData);
    setIsProcessing(false);
  };

  return (
    <div className="checkout-container">
      <h1>Checkout</h1>
      
      {fraudResult && (
        <div className={`fraud-alert ${fraudResult.is_fraud ? 'danger' : 'safe'}`}>
          Risk Score: {(fraudResult.risk_score * 100).toFixed(0)}%
          {fraudResult.risk_factors.length > 0 && (
            <ul>
              {fraudResult.risk_factors.map(factor => (
                <li key={factor}>{factor}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <button 
        onClick={handlePayment} 
        disabled={isProcessing}
      >
        {isProcessing ? 'Processing...' : 'Complete Purchase'}
      </button>
    </div>
  );
}

// Helper functions
function getDeviceFingerprint() {
  // Simple device fingerprint (use FingerprintJS in production)
  return navigator.userAgent + navigator.language;
}

function calculateAccountAge(userId) {
  // Get user registration date from your database
  // Return days since registration
  return 30; // Example
}

async function processPayment(data) {
  // Your payment processing logic
}

async function sendToReviewQueue(data, fraudData) {
  // Send to manual review system
}

export default CheckoutPage;
