import React, { useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';

const useStyles = makeStyles((theme) => ({
  submitButton: {
    position: "fixed",
    margin: theme.spacing(1),
    bottom: 0,
  }
}));


export default function SubmitSelectionButton() {
  const classes = useStyles();

  const submitSelection = async () => {
    // Submit user's stock selection to the server
    const settings = {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'selectedStocks': ['123']
        })
    }
    try {
      const response = await fetch("http://127.0.0.1:5000/post-test", settings)
      if (response.ok) {
        const jsonData = await response.json();
        console.log(jsonData);
      }
    }
    catch (err) {
      console.log('fetch failed', err);
    }
  }

  return (
    <div className={classes.submitButton}>
      <Button variant="contained" color="primary" size='large' onClick={(e) => {submitSelection()}}>
        Create Portfolio
      </Button>
    </div>
  );
}