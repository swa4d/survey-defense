List of flags:
BrowserTimeZone,ComputerLanguage,ComputerTime,ExternalDataReference,PanelID,Q_BallotBoxStuffing,Q_RecaptchaScore,Q_TerminateFlag,Q_TotalDuration,Q_URL,RecipientEmail,RecipientFirstName,RecipientLastName,Referer,UserAgent,ally,ally_no,ally_yes,auto_domAutomation,auto_langsLen,auto_nightmare,auto_phantom,auto_pluginsLen,auto_score,auto_webdriver,captcha_pass,gc,hackathon_setup,ht_backspace_QID584,ht_backspace_QID629,ht_backspace_QID631,ht_backspace_QID690,ht_backspace_QID694,ht_flag_bigPaste_QID584,ht_flag_bigPaste_QID629,ht_flag_bigPaste_QID631,ht_flag_bigPaste_QID690,ht_flag_bigPaste_QID694,ht_flag_lowKeysHighText_QID584,ht_flag_lowKeysHighText_QID629,ht_flag_lowKeysHighText_QID631,ht_flag_lowKeysHighText_QID690,ht_flag_lowKeysHighText_QID694,ht_flag_uniformTiming_QID584,ht_flag_uniformTiming_QID629,ht_flag_uniformTiming_QID631,ht_flag_uniformTiming_QID690,ht_flag_uniformTiming_QID694,ht_inputDurMs_QID584,ht_inputDurMs_QID629,ht_inputDurMs_QID631,ht_inputDurMs_QID690,ht_inputDurMs_QID694,ht_interKeyIqrMs_QID584,ht_interKeyIqrMs_QID629,ht_interKeyIqrMs_QID631,ht_interKeyIqrMs_QID690,ht_interKeyIqrMs_QID694,ht_interKeyMedMs_QID584,ht_interKeyMedMs_QID629,ht_interKeyMedMs_QID631,ht_interKeyMedMs_QID690,ht_interKeyMedMs_QID694,ht_keydown_QID584,ht_keydown_QID629,ht_keydown_QID631,ht_keydown_QID690,ht_keydown_QID694,ht_paste_QID584,ht_paste_QID629,ht_paste_QID631,ht_paste_QID690,ht_paste_QID694,ht_pastedChars_QID584,ht_pastedChars_QID629,ht_pastedChars_QID631,ht_pastedChars_QID690,ht_pastedChars_QID694,ht_totalChars_QID584,ht_totalChars_QID629,ht_totalChars_QID631,ht_totalChars_QID690,ht_totalChars_QID694,regime,regime_dem,regime_nondem,rid,team_code,team_email,token,trade,trade_no,trade_yes,userAgent,welfare,welfare_lazy,welfare_unluckly,gender_First Click,gender_Last Click,gender_Page Submit,gender_Click Count,hispanic_First Click,hispanic_Last Click,hispanic_Page Submit,hispanic_Click Count,race_First Click,race_Last Click,race_Page Submit,race_Click Count

Analysis Strategy:
We are going to use an ensemble of classifiers, each employing their own strategies. Each classifier will return a list of tuples, each containing the identifier of each survey attempt, a vote of if the survey attempt is a bot (0 if not), and a confidence value for this vote (0 if no confidence, and -1 for no interval). We can then decide on a final probability/confidence and a final vote for each survey attempt by isolating each tag, and then creating a weighted sum of votes from each classifier:


For each survey attempt we use the equation Sigma_classifiers = w_c * vote_c * confidence_c


* we can apply weight updates using assured voters, and bootstrapping all other attempts, like full attempts with impossibly low times since they are assured bot attempts.

Classifiers:
Random forest identifying outliers in the following categories:
Identifying attempts with no keystrokes
Identifying questions with no click counts
Identifying attempts with low Q_captcha scores, scaled to indicate confidence
Using auto_pluginsLen & auto_langsLen as a flag
ht_flag_uniformTiming
ht_flag_bigPaste
ht_flag_lowKeysHighText
