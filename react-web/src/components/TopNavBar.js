import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Button from '@material-ui/core/Button';
import NavBarAccountButton from './NavBarAccountButton'
import LoginDialog from './LoginDialog'

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
}));

export default function TopNavBar(props) {
  //This component is the navigation bar on the top of the page
  const classes = useStyles();
  const [isLoginDialogOpen, setLoginDialogOpen] = React.useState(false);

  const handleLoginDialogOpen = () => {
    setLoginDialogOpen(true);
  };

  const handleLoginDialogClose = () => {
    setLoginDialogOpen(false);
  };

  return (
    <div className={classes.root}>
      <AppBar position="static">
        <LoginDialog isOpen={isLoginDialogOpen} handleClose={handleLoginDialogClose} setUserData={props.setUserData}></LoginDialog>
        <Toolbar>
          <Button className={classes.brandButton} color="inherit" size="large">
            AI Asset
          </Button>
          <section className={classes.rightButtons}>
            {
              props.userData.userEmail == undefined
              ? 
                <Button color="inherit" onClick={(e) => {handleLoginDialogOpen()}}>
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