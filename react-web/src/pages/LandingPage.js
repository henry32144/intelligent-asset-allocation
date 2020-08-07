import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
// core components

// Sections for this page

const useStyles = makeStyles((theme) => ({
  masthead: {
    background: "url(../static/landing_img2.jpg)",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    backgroundSize: "cover",
    marginLeft: "auto",
    marginRight: "auto",
    height: "100vh",
    maxHeight: "1600px",
    padding: theme.spacing(0, 2, 0),

  },
  mastheadText: {
    margin: theme.spacing(0, 4, 0),
  },
  mastheadButton: {
    backgroundColor: "#81bc44",
    color: "white"
  },
  mastheadText: {
    color: "white"
  }
}));

export default function LandingPage(props) {
  const classes = useStyles();
  return (
    <div>
      <Grid container className={classes.masthead} justify="center" alignItems="center">
        <Grid item md={8} sm={10} xs={12}>
          <Grid item md={6} sm={8} xs={12} className={classes.mastheadText}>
            <Typography variant="h4" className={classes.mastheadText}>Invest smarter with Hugging Money.</Typography>
            <Typography variant="subtitle2" className={classes.mastheadText}>
              Join Hugging Money and get tools to help you build your own portfolio, without paying the transaction fee, management fee, redemption as like you buy mutual funds.
              </Typography>
            <br />
            <Button
              className={classes.mastheadButton}
              size="large"
              onClick={() => { }}
            >
              Start Now
             </Button>
          </Grid>
        </Grid>
      </Grid>
    </div>
  )
}