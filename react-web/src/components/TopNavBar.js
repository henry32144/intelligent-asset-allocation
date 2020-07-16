import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Button from '@material-ui/core/Button';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  brandButton: {
    marginRight: theme.spacing(2),
  },
  rightButtons: {
    marginLeft: "auto",
  },
}));

export default function TopNavBar() {
  //This component is the navigation bar on the top of the page
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <AppBar position="static">
        <Toolbar>
          <Button className={classes.brandButton} color="inherit" size="large">
            AI Asset
          </Button>
          <section className={classes.rightButtons}>
            <Button className={classes.loginButton} color="inherit" aria-label="Login">
                Login
            </Button>
          </section>
        </Toolbar>
      </AppBar>
    </div>
  );
}