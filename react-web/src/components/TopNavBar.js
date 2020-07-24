import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Button from '@material-ui/core/Button';
import NavBarAccountButton from './NavBarAccountButton'
import LoginDialog from './LoginDialog'
import SignupDialog from './SignupDialog'
import MessageDialog from './MessageDialog'
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import Drawer from '@material-ui/core/Drawer';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  menuButton: {
    marginRight: theme.spacing(2),
  },
  brandButton: {
    marginRight: theme.spacing(2),
  },
  rightButtons: {
    marginLeft: "auto",
    display: 'inline-flex',
  },
}));

export default function TopNavBar(props) {
  //This component is the navigation bar on the top of the page
  const classes = useStyles();
  const [isDrawerOpen, setDrawerOpen] = React.useState(false);
  const [isLoginDialogOpen, setLoginDialogOpen] = React.useState(false);
  const [isSignupDialogOpen, setSignupDialogOpen] = React.useState(false);
  const [isMessageDialogOpen, setMessageDialogOpen] = React.useState(false);
  const [dialogMessage, setDialogMessage] = React.useState("");

  const toogleDrawer = () => {
    setDrawerOpen(!isDrawerOpen);
  };

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
      <Drawer
          className={classes.drawer}
          variant="persistent"
          anchor="left"
          open={isDrawerOpen}
          classes={{
            paper: classes.drawerPaper,
          }}
        >
      </Drawer>
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
          <IconButton 
            edge="start" 
            className={classes.menuButton}
            onClick={toogleDrawer}
            color="inherit" 
            aria-label="menu">
            <MenuIcon />
          </IconButton>
          <Button className={classes.brandButton} color="inherit" size="large">
            AI Asset
          </Button>
          <section className={classes.rightButtons}>
            {
              props.userData.userEmail == undefined
                ?
                <Button color="inherit" onClick={(e) => { handleLoginDialogOpen() }}>
                  Login
                </Button>
                :
                <NavBarAccountButton setUserData={props.setUserData}></NavBarAccountButton>
            }
          </section>
        </Toolbar>
      </AppBar>
    </div>
  );
}