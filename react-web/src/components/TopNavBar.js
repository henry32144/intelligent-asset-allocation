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
  const [isLogin, setSelectedData] = React.useState(props.userData == undefined ? true : false);
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
        <LoginDialog isOpen={isLoginDialogOpen} handleClose={handleLoginDialogClose}></LoginDialog>
        <Toolbar>
          <Button className={classes.brandButton} color="inherit" size="large">
            AI Asset
          </Button>
          <section className={classes.rightButtons}>
            {
              true 
              ? 
                <Button color="inherit" onClick={(e) => {handleLoginDialogOpen()}}>
                  Login
                </Button>
              : 
                <NavBarAccountButton />
            }
          </section>
        </Toolbar>
      </AppBar>
    </div>
  );
}