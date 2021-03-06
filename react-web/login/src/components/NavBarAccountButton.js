import React from 'react';
import Button from '@material-ui/core/Button';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import Grow from '@material-ui/core/Grow';
import Paper from '@material-ui/core/Paper';
import Popper from '@material-ui/core/Popper';
import MenuItem from '@material-ui/core/MenuItem';
import MenuList from '@material-ui/core/MenuList';
import { makeStyles } from '@material-ui/core/styles';
import AccountCircleIcon from '@material-ui/icons/AccountCircle';
import Cookies from 'universal-cookie';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
  },
  accountButton: {
    color: 'inherit'
  },
  popperStyle: {
    zIndex: '1400',
  }
}));

export default function NavBarAccountButton(props) {
  const classes = useStyles();
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    setOpen(false);
  };

  const logoutButtonOnClick = (e) => {
    props.setUserData({
      userId: null,
      userName: null,
      userEmail: null,
    });

    // Remove user data from cookies
    const cookies = new Cookies();
    cookies.remove('userId', { path: '/' });
    cookies.remove('userName', { path: '/' });
    cookies.remove('userEmail', { path: '/' });

    handleClose(e)
    window.location.pathname = './';
  };

  function handleListKeyDown(event) {
    if (event.key === 'Tab') {
      event.preventDefault();
      setOpen(false);
    }
  }

  // return focus to the button when we transitioned from !open -> open
  const prevOpen = React.useRef(open);
  React.useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }

    prevOpen.current = open;
  }, [open]);

  return (
    <div className={classes.root}>
      <div>
        <Button 
          startIcon={<AccountCircleIcon />}
          endIcon={<ExpandMoreIcon />}
          ref={anchorRef}
          aria-controls={open ? 'menu-list-grow' : undefined}
          aria-haspopup="true"
          className={classes.accountButton}
          onClick={handleToggle}
        >
          {
            props.userData.userName == undefined ? 
              "Account"
            : 
            props.userData.userName
          }
        </Button>
        <Popper 
          className={classes.popperStyle} 
          open={open} 
          anchorEl={anchorRef.current} 
          role={undefined} 
          transition 
          disablePortal>
          {({ TransitionProps, placement }) => (
            <Grow
              {...TransitionProps}
              style={{ transformOrigin: placement === 'bottom' ? 'center top' : 'center bottom' }}
            >
              <Paper>
                <ClickAwayListener onClickAway={handleClose}>
                  <MenuList autoFocusItem={open} id="menu-list-grow" onKeyDown={handleListKeyDown}>
                    <MenuItem onClick={logoutButtonOnClick}>Logout</MenuItem>
                  </MenuList>
                </ClickAwayListener>
              </Paper>
            </Grow>
          )}
        </Popper>
      </div>
    </div>
  );
}
