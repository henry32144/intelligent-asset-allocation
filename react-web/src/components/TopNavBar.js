import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Button from '@material-ui/core/Button';
import NavBarAccountButton from './NavBarAccountButton'
import LoginDialog from './LoginDialog'
import SignupDialog from './SignupDialog'
import MessageDialog from './MessageDialog'
import { HOME_PAGE, DASHBOARD_PAGE } from '../Constants';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  brandButton: {
    marginRight: theme.spacing(2),
  },
  rightButtons: {
    marginLeft: "auto",
    display: 'inline-flex',
  },
  iconSize: {
    width: "26px",
    height: "26px",
  },
  accountButtons: {
    display: 'inline-flex',
  },
  dashboardButton: {
    marginRight: theme.spacing(1)
  },
}));

export default function TopNavBar(props) {
  //This component is the navigation bar on the top of the page
  const classes = useStyles();
  const [isLoginDialogOpen, setLoginDialogOpen] = React.useState(false);
  const [isSignupDialogOpen, setSignupDialogOpen] = React.useState(false);
  const [isMessageDialogOpen, setMessageDialogOpen] = React.useState(false);
  const [dialogMessage, setDialogMessage] = React.useState("");

  const handleLoginDialogOpen = () => {
    setLoginDialogOpen(true);
  };

  const handleLoginDialogClose = () => {
    setLoginDialogOpen(false);
  };

  const handleSignupDialogOpen = () => {
    setSignupDialogOpen(true);
  };

  const handleSignupDialogClose = () => {
    setSignupDialogOpen(false);
  };

  const handleMessageDialogOpen = () => {
    setMessageDialogOpen(true);
  };

  const handleMessageDialogClose = () => {
    setMessageDialogOpen(false);
  };

  return (
    <div className={classes.root}>
      <AppBar position="static">
        <LoginDialog
          isOpen={isLoginDialogOpen}
          handleClose={handleLoginDialogClose}
          setUserData={props.setUserData}
          openSignup={handleSignupDialogOpen}>
        </LoginDialog>
        <SignupDialog
          isOpen={isSignupDialogOpen}
          handleClose={handleSignupDialogClose}
          setDialogMessage={setDialogMessage}
          openMessageDialog={handleMessageDialogOpen}
        >
        </SignupDialog>
        <MessageDialog
          isOpen={isMessageDialogOpen}
          handleClose={handleMessageDialogClose}
          message={dialogMessage}
        >
        </MessageDialog>
        <Toolbar>
          <Button
            className={classes.brandButton}
            color="inherit"
            size="large"
            onClick={() => {props.switchPage(HOME_PAGE)}}
            startIcon={
              <img 
                alt=""
                className={classes.iconSize}
                src={'../static/logo26.png'} 
              />
            }
          >
            Hugging Money
          </Button>
          <section className={classes.rightButtons}>
            {
              props.userData.userEmail == undefined
                ?
                <Button color="inherit" onClick={(e) => { handleLoginDialogOpen() }}>
                  Login
                </Button>
                :
                <div className={classes.accountButtons}>
                  <Button 
                    color="inherit" 
                    className={classes.dashboardButton}
                    onClick={() => {props.switchPage(DASHBOARD_PAGE)}}
                  >
                    Dashboard
                  </Button>
                  <NavBarAccountButton
                    setUserData={props.setUserData}
                    userData={props.userData}
                  >
                  </NavBarAccountButton>
                </div>
            }
          </section>
        </Toolbar>
      </AppBar>
    </div>
  );
}