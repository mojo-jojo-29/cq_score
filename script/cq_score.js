total_cq_score = 0; // Initialize the overall CQ score accumulator
 
// If you have a real backend, replace the above with something like:
async function getScoresFromBackend() { // Define an async function to fetch score data from backend
  const res = await fetch('/api/scores'); // Send a GET request to the APIs from where score can be fetched
  if (!res.ok) throw new Error('Failed to fetch scores'); // Throw an error if the response is not OK
  return await res.json(); // Parse and return the JSON body containing all scores
}

// trust score fetch function
async function getTrustScoreFromBackend() { // Define an async function to fetch trust score data from backend
  const res = await fetch('/api/trust_score'); // Send a GET request to the APIs from where trust score can be fetched
  if (!res.ok) throw new Error('Failed to fetch trust score'); // Throw an error if the response is not OK
  return await res.json(); // Parse and return the JSON body containing trust score
}
 
// Load scores from "backend", assign them, then recalculate
async function loadScoresAndRecalculate() { // Define an async function to load and apply scores
    const data = await getScoresFromBackend(); // Await backend scores before proceeding
    const trustScore = await getTrustScoreFromBackend(); // Await backend trust score before proceeding
    // assign scalar scores
    first_name_score = data.first_name_score; // Set first name score from backend
    last_name_score = data.last_name_score; // Set last name score from backend
    dob_score = data.dob_score; // Set date of birth score from backend
    email_score = data.email_score; // Set email score from backend
    phone_score = data.phone_score; // Set phone score from backend
    gender_score = data.gender_score; // Set gender score from backend
    organisation_score = data.organisation_score; // Set organization score from backend
    sport_score = data.sport_score; // Set sport score from backend
    adhar_score = data.adhar_score; // Set Aadhar score from backend
    pan_code_score = data.pan_code_score; // Set PAN code score from backend
    my_people_score = data.my_people_score; // Set "my people" score from backend
    my_places_score = data.my_places_score; // Set "my places" score from backend
    trust_score = trustScore.trust_score; // Set trust score from backend
 
    // assign object scores
    stats_score = { ...data.stats_score }; // Copy stats score object so future changes don't mutate source
    personal_cert_score = { ...data.personal_cert_score }; // Copy personal certificates object
    edu_cert_score = { ...data.edu_cert_score }; // Copy education certificates object
    gold_cert_score = { ...data.gold_cert_score }; // Copy gold certificates object
    silver_cert_score = { ...data.silver_cert_score }; // Copy silver certificates object
    bronze_cert_score = { ...data.bronze_cert_score }; // Copy bronze certificates object
    participation_cert_score = { ...data.participation_cert_score }; // Copy participation certificates object
    coach_exp_score = { ...data.coach_exp_score }; // Copy coaching experience object
    volunteer_exp_score = { ...data.volunteer_exp_score }; // Copy volunteering experience object
 
    // optional streak inputs used in recalculateTotalCQScore()
    daily_count = data.daily_count; // Set number of consecutive daily visits
    missed_days = data.missed_days; // Set number of missed days since last streak
 
    recalculateTotalCQScore(); // Recompute the total CQ score using the latest values
}
 
// Helper function to sum values of an object
function sumObjectValues(obj) { // Define a function that totals numeric values of an object's properties
    return Object.values(obj).reduce((a, b) => a + b, 0); // Convert to array of values and sum them with reduce
}
 
// Function to update total_cq_score based on current keys
function recalculateTotalCQScore() { // Define a function that recomputes the overall CQ score
    const baseScalars = [ // Collect all scalar scores in an array for concise summation
        first_name_score, // First name score
        last_name_score, // Last name score
        dob_score, // Date of birth score
        email_score, // Email score
        phone_score, // Phone score
        gender_score, // Gender score
        organisation_score, // Organization score
        sport_score, // Sport score
        adhar_score, // Aadhar score
        pan_code_score, // PAN code score
        my_people_score, // "My people" score
        my_places_score // "My places" score
    ];
 
    total_cq_score = baseScalars.reduce((sum, v) => sum + v, 0); // Sum all scalar scores to initialize the total
 
    const groupedObjects = [ // List of grouped objects whose values need to be summed
        stats_score, // Stats group
        personal_cert_score, // Personal certificates group
        edu_cert_score, // Education certificates group
        gold_cert_score, // Gold certificates group
        silver_cert_score, // Silver certificates group
        bronze_cert_score, // Bronze certificates group
        participation_cert_score, // Participation certificates group
        coach_exp_score, // Coaching experience group
        volunteer_exp_score // Volunteering experience group
    ];
 
    total_cq_score += groupedObjects // Add the sum of each group's values to the total
        .reduce((sum, obj) => sum + sumObjectValues(obj), 0); // Accumulate sums of all grouped objects
 
    const cappedStreakDays = Math.min(daily_count || 0, 7); // Use up to 7 streak days (default 0 if undefined)
    let daily_visit_score = cappedStreakDays * 5; // Each valid streak day contributes 5 points (max 35)
    if (typeof missed_days === 'number' && missed_days > 0) { // If we have missed days as a positive number
        daily_visit_score = Math.max(0, daily_visit_score - missed_days * 5); // Subtract 5 per missed day, floor at 0
    }
 
    total_cq_score += daily_visit_score; // Add the daily visit component to the total
    total_cq_score = Math.min(total_cq_score, 620); // Enforce the upper bound of 620 on the total score
}